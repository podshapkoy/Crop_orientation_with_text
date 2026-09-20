from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "project"
sys.path.insert(0, str(PROJECT_DIR))

from predict import validate_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Необязательная визуализация готового submission")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_DIR / "test" / "images")
    parser.add_argument("--submission", type=Path, default=PROJECT_DIR / "output" / "submission.csv")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "model_random_15.png")
    args = parser.parse_args()

    image_names = sorted(path.name for path in args.test_dir.glob("*.png"))
    validate_submission(args.submission, image_names)

    with args.submission.open("r", encoding="utf-8", newline="") as file:
        predictions = {
            row["image_id"]: float(row["p_180"])
            for row in csv.DictReader(file)
        }

    rng = random.Random(args.seed)
    selected_names = rng.sample(image_names, min(args.count, len(image_names)))
    columns = 5
    rows = (len(selected_names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows))
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for axis, image_name in zip(axes, selected_names):
        with Image.open(args.test_dir / image_name) as image:
            axis.imshow(image)
        probability = predictions[image_name]
        label = "180°" if probability >= 0.5 else "0°"
        axis.set_title(
            f"{image_name}\nПредсказано: {label}\nP(180°) = {probability:.3f}",
            fontsize=10,
        )
        axis.axis("off")

    for axis in axes[len(selected_names):]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Сохранено -> {args.output}")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
