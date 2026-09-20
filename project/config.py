import os
import random
from pathlib import Path

import numpy as np

"""структура директории и пути"""
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
SYNTH_DIR = DATA_DIR / "synthetic"
SYNTH_IMAGES = SYNTH_DIR / "images"
SYNTH_MANIFEST = SYNTH_DIR / "manifest.csv"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"

TEST_IMAGES = ROOT / "test" / "images"
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "output"

MODEL_NAME = "PP-LCNet_x1_0_textline_ori"
MODEL_DIR = ROOT / "models" / MODEL_NAME
CALIBRATION_PATH = CHECKPOINTS_DIR / "calibration.json"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

"""параметры датасета и валидации"""
EXPECTED_SYNTHETIC_IMAGE_COUNT = 20_000
EXPECTED_TEST_IMAGE_COUNT = 20_000
TEST_FILENAME_TEMPLATE = "test_{index:05d}.png"

# фиксация случайности
SEED = 42

# сплит синтетики для обучения калибратора
SPLIT_VERSION = "group_hash_v1"
DEVELOPMENT_FRACTION = 0.80 # для выбора лучшего метода калибровки
CALIBRATION_FRACTION = 0.10 # для обучения параметров
VALIDATION_FRACTION = 0.10
INFERENCE_BATCH_SIZE = 128
CPU_THREADS = 4
CALIBRATION_SAMPLE_SIZE = 8_000

"""настройки синтетической генерации"""
N_SYNTHETIC_RU = 10_000
N_SYNTHETIC_EN = 10_000
N_BACKGROUNDS = 3_000 #  фото-фоны из COCO

RU_DOMAIN_WEIGHTS = {
    "ru_household": 0.75,
    "ru_price_tag": 0.25,
}

EN_DOMAIN_WEIGHTS = {
    "en_brand": 0.25,
    "en_tech": 0.25,
    "en_comic": 0.15,
    "en_book": 0.20,
    "en_price_tag": 0.15,
}

FONT_CATEGORY_WEIGHTS = {
    "sans": 0.34,
    "serif": 0.24,
    "mono": 0.14,
    "condensed": 0.14,
    "comic": 0.14,
}

# искривления и повороты
TEXT_MODE_WEIGHTS = {"straight": 0.60, "diagonal": 0.32, "curved": 0.08}
DIAGONAL_ANGLE_RANGE = (-18.0, 18.0)

"""деградация изображений (имитация плохих фото)"""
LIGHT_BG_PROB = 0.15
PHOTO_BG_PROB = 0.55
GRADIENT_BG_PROB = 0.20
SOLID_BG_PROB = 0.25
JPEG_RECOMPRESS_PROB = 0.35 # артефакты сжатия картинки
JPEG_QUALITY_RANGE = (25, 75)
PERSPECTIVE_PROB = 0.25 # фотографии текста под углом
MOTION_BLUR_PROB = 0.15 # смазанность из-за дрожания рук

"""ссылки на данные"""
TRAIN_DATA_URL = "https://disk.yandex.ru/d/HyHIzF7JIi9DIA"
TEST_DATA_URL = "https://disk.360.yandex.ru/d/ha0Q70T7ie-0_w"

"""фиксация сидов"""
def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
