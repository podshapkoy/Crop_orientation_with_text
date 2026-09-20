from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from config import (
    CALIBRATION_PATH,
    EXPECTED_TEST_IMAGE_COUNT,
    INFERENCE_BATCH_SIZE,
    MODEL_NAME,
    SUBMISSION_PATH,
    TEST_FILENAME_TEMPLATE,
    TEST_IMAGES,
)
from orientation import OrientationClassifier, temperature_scale, isotonic_scale


def expected_test_names(count: int = EXPECTED_TEST_IMAGE_COUNT) -> list[str]:
    return [TEST_FILENAME_TEMPLATE.format(index=i) for i in range(count)]


def find_test_paths(test_dir: Path = TEST_IMAGES) -> list[Path]:
    names = expected_test_names()
    return [test_dir / name for name in names]


def load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    with path.open("r", encoding="utf-8") as file:
        calibration = json.load(file)
    if calibration.get("model_name") != MODEL_NAME:
        raise ValueError("калибровка для другой модели")
    method = calibration.get("method", "temperature")
    if method == "temperature" and float(calibration["temperature"]) <= 0:
        raise ValueError("temperature не положительная")
    if method == "isotonic":
        x, y = calibration.get("isotonic_x", []), calibration.get("isotonic_y", [])
        if not x or len(x) != len(y):
            raise ValueError("некорректные isotonic thresholds")
    return calibration


def validate_submission(path: Path, expected_names: Sequence[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    if not rows or rows[0] != ["image_id", "p_180"]:
        raise ValueError("неправильный заголоок")
    data = rows[1:]
    if [row[0] for row in data] != list(expected_names):
        raise ValueError("неправильный image_id")
    for row in data:
        if len(row) != 2 or not 0 <= float(row[1]) <= 1:
            raise ValueError("неправильный p_180")


def write_submission(
    output_path: Path,
    image_names: Sequence[str],
    probabilities: Sequence[float],
) -> None:
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) != len(image_names) or not np.all(np.isfinite(probabilities)):
        raise ValueError("неправильные значение вероятности")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["image_id", "p_180"])
        for name, probability in zip(image_names, probabilities):
            writer.writerow([name, f"{float(probability):.12f}"])
    validate_submission(temporary_path, image_names)
    os.replace(temporary_path, output_path)


def predict(
    test_dir: Path = TEST_IMAGES,
    calibration_path: Path = CALIBRATION_PATH,
    output_path: Path = SUBMISSION_PATH,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> Path:
    calibration = load_calibration(calibration_path)
    paths = find_test_paths(test_dir)
    classifier = OrientationClassifier(batch_size=batch_size)
    _, _, tta = classifier.predict_pairs(paths, description="test")
    method = calibration.get("method", "temperature")
    if method == "isotonic":
        probabilities = isotonic_scale(tta, calibration["isotonic_x"], calibration["isotonic_y"])
    elif method == "temperature":
        probabilities = temperature_scale(tta, float(calibration["temperature"]))
    else:
        probabilities = tta
    names = [path.stem for path in paths]
    write_submission(output_path, names, probabilities)
    print(f"Готово: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="создается submission.csv")
    parser.add_argument("--test-dir", type=Path, default=TEST_IMAGES)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    parser.add_argument("--batch-size", type=int, default=INFERENCE_BATCH_SIZE)
    args = parser.parse_args()
    predict(args.test_dir, args.calibration, args.output, args.batch_size)


if __name__ == "__main__":
    main()
