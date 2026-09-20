"""
модуль подбора параметров калибровки вероятностей

Из-за целевой метрики необходимо выдавать максимально честную вероятность, и нужна калибровка

- Разбиваю датасет на три фолда: calibration (фит параметров), development (выбор лучшего метода)
и validation
- Сравниваю два подхода: параметрический (Temperature Scaling) и непараметрический (Isotonic Regression)
- Сохраняю параметры победившего метода для использования на инференсе

Также сделана защита от утечки через криптографическое хеширование
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from sklearn.isotonic import IsotonicRegression

from config import (
    CALIBRATION_FRACTION,
    CALIBRATION_PATH,
    CALIBRATION_SAMPLE_SIZE,
    DEVELOPMENT_FRACTION,
    EXPECTED_SYNTHETIC_IMAGE_COUNT,
    INFERENCE_BATCH_SIZE,
    MODEL_NAME,
    SEED,
    SPLIT_VERSION,
    SYNTH_IMAGES,
    SYNTH_MANIFEST,
)
from orientation import (
    OrientationClassifier,
    binary_metrics,
    brier_score,
    fit_temperature,
    isotonic_scale,
    temperature_scale,
)

SPLIT_NAMES = ("development", "calibration", "validation")


def split_number(value: str) -> float:
    """
    генерируем псевдослучайное число от 0 до 1, чтобы воспроизводимость
    разбиения датасета была без привязки к конкретному генератору
    """
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def group_key(record: Mapping[str, str]) -> str:
    """определяем ключ группы, чтобы похожие изображения не попали в разные сплиты"""
    group = record.get("source_group") or record.get("group")
    return group if group and group.startswith("bg_") else record["filename"]


def assign_split(key: str) -> str:
    value = split_number(f"{SPLIT_VERSION}:{SEED}:{key}")
    if value < DEVELOPMENT_FRACTION:
        return "development"
    if value < DEVELOPMENT_FRACTION + CALIBRATION_FRACTION:
        return "calibration"
    return "validation"


def read_manifest(
    manifest_path: Path = SYNTH_MANIFEST,
    images_dir: Path = SYNTH_IMAGES,
) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))

    if len(records) != EXPECTED_SYNTHETIC_IMAGE_COUNT:
        raise ValueError(f"в manifest должно быть {EXPECTED_SYNTHETIC_IMAGE_COUNT} строк")
    if not all((images_dir / record["filename"]).is_file() for record in records):
        raise ValueError("в manifest не все найдено")
    return records


def split_records(records: Iterable[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    splits = {name: [] for name in SPLIT_NAMES}
    for record in records:
        splits[assign_split(group_key(record))].append(record)
    return splits


def make_pairs(original: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    так как синтетические данные изначально сгенерировала горизонтальном формате(под углом 0°),
    нужно еще добавить перевернутые версии(под углом 180°), инвертируя вероятности
    """
    probabilities = np.concatenate((original, 1.0 - original))
    labels = np.concatenate((np.zeros(len(original)), np.ones(len(original))))
    return probabilities, labels


def metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    result = binary_metrics(probabilities, labels)
    result["brier"] = brier_score(probabilities, labels)
    return result


def calibrate(
    manifest_path: Path = SYNTH_MANIFEST,
    images_dir: Path = SYNTH_IMAGES,
    output_path: Path = CALIBRATION_PATH,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> dict:
    records = read_manifest(manifest_path, images_dir)
    records = sorted(
        records,
        key=lambda record: split_number(f"sample:{SEED}:{record['filename']}"),
    )[:CALIBRATION_SAMPLE_SIZE]
    splits = split_records(records)
    print("исходных картинок:", {name: len(value) for name, value in splits.items()})

    classifier = OrientationClassifier(batch_size=batch_size)
    paths = [images_dir / record["filename"] for record in records]
    _, _, tta = classifier.predict_pairs(paths, description="calibration")

    positions = {record["filename"]: index for index, record in enumerate(records)}
    split_indices = {
        name: np.asarray([positions[record["filename"]] for record in values])
        for name, values in splits.items()
    }

    calibration_p, calibration_y = make_pairs(tta[split_indices["calibration"]])
    development_p, development_y = make_pairs(tta[split_indices["development"]])

    raw_brier = brier_score(development_p, development_y)

    # обучаем temperature scaling
    temperature = fit_temperature(calibration_p, calibration_y)
    temperature_brier = brier_score(
        temperature_scale(development_p, temperature), development_y
    )

    # обучаем isotonic regression
    isotonic_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic_model.fit(calibration_p, calibration_y)
    x_thresholds = isotonic_model.X_thresholds_.tolist()
    y_thresholds = isotonic_model.y_thresholds_.tolist()
    isotonic_brier = brier_score(
        isotonic_scale(development_p, x_thresholds, y_thresholds), development_y
    )

    # выбираем лучший метод по development, чтобы не переобучиться
    candidates = {"identity": raw_brier, "temperature": temperature_brier, "isotonic": isotonic_brier}
    method = min(candidates, key=candidates.get)
    print("development brier по методам:", candidates, "-> выбран:", method)

    def apply_calibration(probabilities: np.ndarray) -> np.ndarray:
        if method == "temperature":
            return temperature_scale(probabilities, temperature)
        if method == "isotonic":
            return isotonic_scale(probabilities, x_thresholds, y_thresholds)
        return probabilities

    all_metrics = {}
    for name, indices in split_indices.items():
        probabilities, labels = make_pairs(tta[indices])
        all_metrics[name] = {
            "tta": metrics(probabilities, labels),
            "calibrated": metrics(apply_calibration(probabilities), labels),
        }
        print(name, all_metrics[name])
    # проверяю на адекватность модели перед сохранением
    validation_p = tta[split_indices["validation"]]
    calibrated_validation = apply_calibration(validation_p)
    if np.median(calibrated_validation) >= 0.5:
        raise RuntimeError("перепутана полярность p_180")
    if all_metrics["validation"]["calibrated"]["accuracy"] < 0.80:
        raise RuntimeError("плохие данные модели")
    # сохраняю артефакт(он будет подхвачен на этапе инференса)
    artifact = {
        "schema_version": 2,
        "model_name": MODEL_NAME,
        "method": method,
        "temperature": temperature,
        "isotonic_x": x_thresholds,
        "isotonic_y": y_thresholds,
        "seed": SEED,
        "split_version": SPLIT_VERSION,
        "sample_size": CALIBRATION_SAMPLE_SIZE,
        "split_counts": {name: len(value) for name, value in splits.items()},
        "metrics": all_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"калибровка: {output_path}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=SYNTH_MANIFEST)
    parser.add_argument("--images", type=Path, default=SYNTH_IMAGES)
    parser.add_argument("--output", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--batch-size", type=int, default=INFERENCE_BATCH_SIZE)
    args = parser.parse_args()
    calibrate(args.manifest, args.images, args.output, args.batch_size)


if __name__ == "__main__":
    main()
