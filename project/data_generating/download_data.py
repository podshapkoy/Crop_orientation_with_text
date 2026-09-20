from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable

import requests

from config import (
    DATA_DIR,
    EXPECTED_SYNTHETIC_IMAGE_COUNT,
    EXPECTED_TEST_IMAGE_COUNT,
    ROOT,
    TEST_DATA_URL,
    TEST_FILENAME_TEMPLATE,
    TRAIN_DATA_URL,
)

YANDEX_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
MARKERS_DIR = DATA_DIR / ".markers"


def get_direct_link(public_url: str, attempts: int = 5) -> str:
    for attempt in range(attempts):
        try:
            response = requests.get(
                YANDEX_API,
                params={"public_key": public_url},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["href"]
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("нет ссылки")


def download_file(url: str, path: Path) -> None:
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with path.open("wb") as file:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r{downloaded / 1e6:.0f}/{total / 1e6:.0f} МБ", end="")
    print()


def find_directory(root: Path, suffix: str) -> Path:
    suffix_parts = Path(suffix).parts
    for path in root.rglob(suffix_parts[-1]):
        if path.is_dir() and path.relative_to(root).parts[-len(suffix_parts):] == suffix_parts:
            return path
    raise ValueError(f"в архиве нет: {suffix}")


def validate_synthetic(path: Path) -> None:
    image_names = {image.name for image in (path / "images").glob("*.png")}
    with (path / "manifest.csv").open("r", encoding="utf-8", newline="") as file:
        manifest_names = {row["filename"] for row in csv.DictReader(file)}
    if len(image_names) != EXPECTED_SYNTHETIC_IMAGE_COUNT or image_names != manifest_names:
        raise ValueError("в обучающем файле должно быть 20к картинок и manifest")


def validate_test(path: Path) -> None:
    actual = sorted(image.name for image in path.glob("*.png"))
    expected = [TEST_FILENAME_TEMPLATE.format(index=i) for i in range(EXPECTED_TEST_IMAGE_COUNT)]
    if actual != expected:
        raise ValueError("в тестовом должно быть test_00000.png–test_19999.png")


def download_dataset(
    public_url: str,
    target: Path,
    archive_suffix: str,
    marker_name: str,
    validator: Callable[[Path], None],
    force: bool,
) -> None:
    if target.exists() and not force:
        validator(target)
        print(f"[{marker_name}] уже скачано: {target}")
        return

    direct_url = get_direct_link(public_url)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        archive_path = temporary / "data.zip"
        unpacked = temporary / "unpacked"
        unpacked.mkdir()

        print(f"[{marker_name}] скачивание")
        download_file(direct_url, archive_path)
        print(f"[{marker_name}] распаковка")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(unpacked)

        source = find_directory(unpacked, archive_suffix)
        validator(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(source), str(target))

    MARKERS_DIR.mkdir(parents=True, exist_ok=True)
    (MARKERS_DIR / marker_name).touch()
    print(f"[{marker_name}] готово: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("train", "test"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.only in (None, "train"):
        download_dataset(
            TRAIN_DATA_URL,
            DATA_DIR / "synthetic",
            "synthetic",
            "train_downloaded",
            validate_synthetic,
            args.force,
        )
    if args.only in (None, "test"):
        download_dataset(
            TEST_DATA_URL,
            ROOT / "test" / "images",
            "test/images",
            "test_downloaded",
            validate_test,
            args.force,
        )


if __name__ == "__main__":
    main()
