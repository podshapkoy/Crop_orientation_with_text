from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

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
    temperature_scale,
)

SPLIT_NAMES = ("development", "calibration", "validation")


def split_number(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def group_key(record: Mapping[str, str]) -> str:
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
    candidate_temperature = fit_temperature(calibration_p, calibration_y)

    development_p, development_y = make_pairs(tta[split_indices["development"]])
    raw_brier = brier_score(development_p, development_y)
    candidate_brier = brier_score(
        temperature_scale(development_p, candidate_temperature), development_y
    )
    temperature = candidate_temperature if candidate_brier < raw_brier else 1.0

    all_metrics = {}
    for name, indices in split_indices.items():
        probabilities, labels = make_pairs(tta[indices])
        all_metrics[name] = {
            "tta": metrics(probabilities, labels),
            "calibrated": metrics(temperature_scale(probabilities, temperature), labels),
        }
        print(name, all_metrics[name])

    validation_p = tta[split_indices["validation"]]
    calibrated_validation = temperature_scale(validation_p, temperature)
    if np.median(calibrated_validation) >= 0.5:
        raise RuntimeError("перепутана полярность p_180")
    if all_metrics["validation"]["calibrated"]["accuracy"] < 0.80:
        raise RuntimeError("плохие данные модели")

    artifact = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "temperature": temperature,
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
