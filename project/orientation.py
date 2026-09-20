"""основная логика модели и метрик"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from config import CPU_THREADS, INFERENCE_BATCH_SIZE, MODEL_DIR, MODEL_NAME

LABELS = {"0_degree", "180_degree"}
EPS = 1e-6


def check_model_dir(model_dir: Path = MODEL_DIR) -> None:
    """проверяем наличие файлов модели PaddleOCR в директории"""
    required = ("inference.json", "inference.pdiparams", "inference.yml")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"пустая папка: {missing}")


def verify_model_files(model_dir: Path = MODEL_DIR) -> dict[str, str]:
    check_model_dir(model_dir)
    return {name: "local" for name in ("inference.json", "inference.pdiparams", "inference.yml")}


def extract_p180(result: Mapping) -> float:
    """
    извлекаем вероятность перевернутости текста на 180 градусов и
    нормирует скоры, чтобы сумма вероятностей всегда была равна 1
    """
    labels = [str(label) for label in result["label_names"]]
    scores = np.asarray(result["scores"], dtype=float).reshape(-1)
    if set(labels) != LABELS or len(labels) != 2 or len(scores) != 2:
        raise ValueError(f"непонятные классы: {labels}")
    # если сырые выходы модели не дают в сумме 1
    scores = scores / scores.sum()
    return float(scores[labels.index("180_degree")])


def rotation_tta_probability(
        original: Sequence[float] | np.ndarray,
        rotated: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    вычисляем итоговую вероятность поворота на 180°, усредняя предсказания для
    исходной картинки и картинки, перевернутой на 180°.
    Формула: 0.5 * (p_180_orig + (1 - p_180_rotated_image))
    """
    original = np.asarray(original, dtype=float)
    rotated = np.asarray(rotated, dtype=float)
    return 0.5 * (original + 1.0 - rotated)

def isotonic_scale(
    probabilities: Sequence[float] | np.ndarray,
    x_thresholds: Sequence[float],
    y_thresholds: Sequence[float],
) -> np.ndarray:
    """Кусочно-линейная калибровка по точкам изотонической регрессии.
    Не требует sklearn на инференсе — только np.interp"""
    probabilities = np.asarray(probabilities, dtype=float)
    return np.interp(probabilities, x_thresholds, y_thresholds)

def temperature_scale(
        probabilities: Sequence[float] | np.ndarray,
        temperature: float,
) -> np.ndarray:
    """
    применяем Temperature Scaling для смягчения вероятности
    (подбирается на валидационной выборке)
    """
    probabilities = np.asarray(probabilities, dtype=float)
    # защищаем от логарифма нуля/единицы
    probabilities = np.clip(probabilities, EPS, 1.0 - EPS)
    # переходим от вероятностей к логитам
    logits = np.log(probabilities / (1.0 - probabilities))
    # масштабируем логиты и делаем возврат к вероятностям через сигмоиду
    return 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -60, 60)))


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.geomspace(0.25, 4.0, 100)
    scores = [brier_score(temperature_scale(probabilities, t), labels) for t in candidates]
    return float(candidates[int(np.argmin(scores))])


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """основные метрики качества"""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    return {
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
        "brier": brier_score(probabilities, labels),
        "one_minus_brier": 1.0 - brier_score(probabilities, labels),
    }


class OrientationClassifier:
    """
    обертка над моделью PaddleOCR
    """

    def __init__(
            self,
            model_dir: Path = MODEL_DIR,
            batch_size: int = INFERENCE_BATCH_SIZE,
    ) -> None:
        check_model_dir(model_dir)
        from paddleocr import TextLineOrientationClassification

        self.batch_size = batch_size
        self.model = TextLineOrientationClassification(
            model_dir=str(model_dir),
            model_name=MODEL_NAME,
            topk=2,
            device="cpu",
            cpu_threads=CPU_THREADS,
            enable_mkldnn=True,
        )

    def predict_raw(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """сырой инференс модели на батче изображений"""
        results = self.model.predict(
            input=list(images),
            batch_size=min(self.batch_size, len(images)),
        )
        return np.asarray([extract_p180(result) for result in results], dtype=float)

    def predict_pairs(
            self,
            paths: Iterable[Path],
            show_progress: bool = True,
            description: str = "orientation",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Для каждого изображения подает в модель оригинал и копию, повернутую на 180°.
        Возвращает вероятности для оригиналов, для повернутых и усредненный TTA
        """
        paths = list(paths)
        original_p = []
        rotated_p = []
        starts = range(0, len(paths), self.batch_size)
        iterator = tqdm(starts, desc=description, unit="batch", disable=not show_progress)

        for start in iterator:
            batch_paths = paths[start:start + self.batch_size]
            original_images = []
            rotated_images = []
            for path in batch_paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"не считалась картинка: {path}")
                original_images.append(image)
                rotated_images.append(cv2.rotate(image, cv2.ROTATE_180))

            original_p.extend(self.predict_raw(original_images))
            rotated_p.extend(self.predict_raw(rotated_images))

        original_p = np.asarray(original_p)
        rotated_p = np.asarray(rotated_p)
        #  итоговые предсказания (TTA)
        tta_p = rotation_tta_probability(original_p, rotated_p)
        return original_p, rotated_p, tta_p


if __name__ == "__main__":
    check_model_dir()
    print(f"Модель найдена: {MODEL_DIR}")
