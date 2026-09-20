

import io
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from datasets import load_dataset
from tqdm import tqdm

from config import (
    SEED,
    N_SYNTHETIC_RU,
    N_SYNTHETIC_EN,
    N_BACKGROUNDS,
    SYNTH_DIR,
    SYNTH_IMAGES,
    BACKGROUNDS_DIR,
    TEXT_MODE_WEIGHTS,
    RU_DOMAIN_WEIGHTS,
    EN_DOMAIN_WEIGHTS,
    FONT_CATEGORY_WEIGHTS,
    LIGHT_BG_PROB,
    DIAGONAL_ANGLE_RANGE,
    PHOTO_BG_PROB,
    GRADIENT_BG_PROB,
    JPEG_RECOMPRESS_PROB,
    JPEG_QUALITY_RANGE,
    PERSPECTIVE_PROB,
    MOTION_BLUR_PROB,
    set_seed,
)

RU_WORDS = [
    "продам", "куплю", "отдам", "новый", "состояние", "хорошее", "отличное",
    "доставка", "самовывоз", "торг", "цена", "звонить", "только", "срочно",
    "гараж", "квартира", "комната", "дом", "участок", "телефон", "смартфон",
    "ноутбук", "телевизор", "холодильник", "машина", "диван", "кресло",
    "стол", "стул", "шкаф", "куртка", "платье", "туфли", "кроссовки", "детский",
    "коляска", "велосипед", "самокат", "инструмент", "дрель", "запчасти",
    "оригинал", "гарантия", "чек", "документы", "владелец", "пробег",
    "коробка", "автомат", "бензин", "дизель", "москва", "казань", "екатеринбург",
]

EN_CLOTHING_BRANDS = [
    "NIKE", "ADIDAS", "PUMA", "REEBOK", "ZARA", "GUCCI", "LEVIS", "VANS",
    "CONVERSE", "LACOSTE", "UNIQLO", "TOMMY HILFIGER", "CALVIN KLEIN",
    "NEW BALANCE", "THE NORTH FACE", "UNDER ARMOUR", "H AND M", "FILA",
]

EN_TECH_WORDS = [
    "APPLE", "SAMSUNG", "SONY", "XIAOMI", "HUAWEI", "IPHONE", "MACBOOK",
    "ANDROID", "BLUETOOTH", "WIRELESS", "CHARGER", "LAPTOP", "TABLET",
    "CAMERA", "SMARTWATCH", "HEADPHONES", "PROCESSOR", "MEMORY", "STORAGE",
    "WARRANTY", "ORIGINAL", "NEW", "USED", "SALE", "DISCOUNT",
]

EN_COMIC_WORDS = [
    "POW", "BAM", "WOW", "BOOM", "BANG", "CRASH", "ZAP", "WHAM", "OOF",
    "ALERT", "WARNING", "DANGER", "STOP", "GO", "WIN", "EPIC", "SUPER", "HERO",
]

EN_BOOK_PHRASES = [
    "Chapter One", "The End", "Once upon a time", "To be continued",
    "In the beginning", "A long time ago", "The story begins",
    "Table of Contents", "Author's Note", "Part Two",
]

EN_GENERIC_WORDS = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello",
    "world", "time", "life", "story", "light", "shadow", "river", "mountain",
    "silence", "journey", "morning", "evening", "letter", "garden", "window",
    "quiet", "distant", "forgotten", "ancient", "wonder", "secret", "island",
]


def random_ru_household():
    n = random.randint(1, 4)
    text = " ".join(random.choices(RU_WORDS, k=n))
    if random.random() < 0.2:
        text += f" {random.randint(1, 999)}"
    if random.random() < 0.1:
        text = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(10, 99)}"
    mode = random.choice(["normal", "upper", "lower", "title"])
    if mode == "upper":
        text = text.upper()
    elif mode == "lower":
        text = text.lower()
    elif mode == "title":
        text = text.title()
    return text


def random_ru_price_tag():
    templates = [
        lambda: f"ЦЕНА {random.randint(100, 99999)}",
        lambda: f"СКИДКА {random.randint(10, 70)}%",
        lambda: f"ТОРГ УМЕСТЕН",
        lambda: f"-{random.randint(10, 70)}%",
        lambda: f"{random.randint(100, 99999)} РУБ",
    ]
    return random.choice(templates)()


def random_en_brand():
    return random.choice(EN_CLOTHING_BRANDS)


def random_en_tech():
    n = random.randint(1, 2)
    text = " ".join(random.choices(EN_TECH_WORDS, k=n))
    if random.random() < 0.3:
        text += f" {random.choice(['PRO', 'MAX', 'MINI', 'LITE', 'PLUS'])}"
    return text


def random_en_comic():
    n = random.randint(1, 2)
    text = " ".join(random.choices(EN_COMIC_WORDS, k=n))
    if random.random() < 0.5:
        text += "!" * random.randint(1, 3)
    return text.upper()


def random_en_book():
    if random.random() < 0.5:
        n = random.randint(2, 5)
        return " ".join(random.choices(EN_GENERIC_WORDS, k=n)).capitalize()
    return random.choice(EN_BOOK_PHRASES)


def random_en_price_tag():
    templates = [
        lambda: f"${random.randint(1, 999)}.{random.randint(0, 99):02d}",
        lambda: f"{random.randint(5, 90)}% OFF",
        lambda: f"-{random.randint(10, 70)}%",
        lambda: "SALE",
        lambda: f"ONLY ${random.randint(1, 999)}",
    ]
    return random.choice(templates)()


RU_DOMAIN_GENERATORS = {
    "ru_household": random_ru_household,
    "ru_price_tag": random_ru_price_tag,
}

EN_DOMAIN_GENERATORS = {
    "en_brand": random_en_brand,
    "en_tech": random_en_tech,
    "en_comic": random_en_comic,
    "en_book": random_en_book,
    "en_price_tag": random_en_price_tag,
}


def random_ru_text():
    domain = random.choices(list(RU_DOMAIN_WEIGHTS.keys()), weights=list(RU_DOMAIN_WEIGHTS.values()))[0]
    return RU_DOMAIN_GENERATORS[domain](), domain


def random_en_text():
    domain = random.choices(list(EN_DOMAIN_WEIGHTS.keys()), weights=list(EN_DOMAIN_WEIGHTS.values()))[0]
    return EN_DOMAIN_GENERATORS[domain](), domain


def style_params_for_domain(domain):
    if domain == "en_comic":
        return {"font_category": "comic", "force_bold_stroke": True,
                 "bright_colors": True, "prefer_paper_bg": False, "draw_box": False}
    if domain == "en_book":
        return {"font_category": "serif", "force_bold_stroke": False,
                 "bright_colors": False, "prefer_paper_bg": True, "draw_box": False}
    if domain in ("en_price_tag", "ru_price_tag"):
        return {"font_category": random.choice(["mono", "condensed"]),
                 "force_bold_stroke": False, "bright_colors": True,
                 "prefer_paper_bg": False, "draw_box": True}
    return {"font_category": None, "force_bold_stroke": False,
             "bright_colors": False, "prefer_paper_bg": False, "draw_box": False}

FONT_CATALOG = [
    ("DejaVu Sans Mono", "mono"),
    ("DejaVu Sans Condensed", "condensed"),
    ("DejaVu Serif Condensed", "condensed"),
    ("DejaVu Sans", "sans"),
    ("DejaVu Serif", "serif"),
    ("Liberation Mono", "mono"),
    ("Liberation Sans", "sans"),
    ("Liberation Serif", "serif"),
    ("FreeMono", "mono"),
    ("FreeSans", "sans"),
    ("FreeSerif", "serif"),
    ("Noto Sans Mono", "mono"),
    ("Noto Sans", "sans"),
    ("Noto Serif", "serif"),
    ("Carlito", "sans"),
    ("Caladea", "serif"),
    ("Comic Neue", "comic"),
]

_FONT_CMAP_CACHE = {}


def _get_cmap_chars(font_path: str) -> set:
    if font_path not in _FONT_CMAP_CACHE:
        try:
            from fontTools.ttLib import TTFont
            tt = TTFont(font_path, lazy=True, fontNumber=0)
            cmap = tt.getBestCmap() or {}
            _FONT_CMAP_CACHE[font_path] = set(cmap.keys())
        except Exception:
            _FONT_CMAP_CACHE[font_path] = set()
    return _FONT_CMAP_CACHE[font_path]


def font_can_render(font_path: str, text: str) -> bool:
    chars = _get_cmap_chars(font_path)
    if not chars:
        return False
    return all(ord(ch) in chars for ch in text if not ch.isspace())


def get_fonts():
    import matplotlib.font_manager as fm
    by_category = {}
    seen_paths = set()
    for f in sorted(fm.fontManager.ttflist, key=lambda item: (item.name, item.fname)):
        path = f.fname
        if path in seen_paths or not Path(path).exists():
            continue
        for pattern, category in FONT_CATALOG:
            if pattern.lower() in f.name.lower():
                by_category.setdefault(category, []).append(path)
                seen_paths.add(path)
                break

    for category in by_category:
        by_category[category].sort()

    if not by_category:
        raise RuntimeError(
            "нет таких шрифтов"
        )

    summary = ", ".join(f"{k}:{len(v)}" for k, v in by_category.items())
    print(f"найдены шрифты по категориям -> {summary}")
    return by_category


_FALLBACK_FONT_PATH = None


def _fallback_font_path() -> str:
    global _FALLBACK_FONT_PATH
    if _FALLBACK_FONT_PATH is None:
        import matplotlib.font_manager as fm
        _FALLBACK_FONT_PATH = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
    return _FALLBACK_FONT_PATH


def pick_font(by_category, text, forced_category=None):
    categories = [forced_category] if forced_category else list(FONT_CATEGORY_WEIGHTS.keys())
    weights = [1.0] if forced_category else [FONT_CATEGORY_WEIGHTS[c] for c in categories]

    tried = set()
    for _ in range(20):
        category = random.choices(categories, weights=weights)[0]
        candidates = [p for p in by_category.get(category, []) if p not in tried]
        random.shuffle(candidates)
        for path in candidates:
            tried.add(path)
            if font_can_render(path, text):
                return path, category

    return _fallback_font_path(), (forced_category or "sans")

def download_backgrounds():
    BACKGROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(BACKGROUNDS_DIR.glob("*.png"))
    if len(existing) >= N_BACKGROUNDS:
        print("фоны есть")
        return
    print("скачивание фонов...")
    dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)
    count = len(existing)
    for sample in tqdm(dataset, total=N_BACKGROUNDS):
        if count >= N_BACKGROUNDS:
            break
        image = sample["image"].convert("RGB")
        if min(image.size) < 128:
            continue
        image.save(BACKGROUNDS_DIR / f"bg_{count:05d}.png", quality=90)
        count += 1
    print(f"{count} фоновых картинок")


_BACKGROUND_FILES = None


def get_background_files():
    global _BACKGROUND_FILES
    if _BACKGROUND_FILES is None:
        _BACKGROUND_FILES = sorted(BACKGROUNDS_DIR.glob("*.png"))
    return _BACKGROUND_FILES


def make_photo_or_solid_background(width, height):
    r = random.random()
    if r < PHOTO_BG_PROB and get_background_files():
        path = random.choice(get_background_files())
        bg = Image.open(path).convert("RGB")
        bw, bh = bg.size
        scale = max(width / bw, height / bh, 1.0) * random.uniform(1.0, 1.5)
        bg = bg.resize((max(1, int(bw * scale)), max(1, int(bh * scale))))
        bw, bh = bg.size
        x0 = random.randint(0, max(0, bw - width))
        y0 = random.randint(0, max(0, bh - height))
        return bg.crop((x0, y0, x0 + width, y0 + height)), path.stem
    if r < PHOTO_BG_PROB + GRADIENT_BG_PROB:
        c1 = np.array([random.randint(0, 255) for _ in range(3)])
        c2 = np.array([random.randint(0, 255) for _ in range(3)])
        x = np.linspace(0, 1, width)[:, None]
        gradient = np.tile(c1 * (1 - x) + c2 * x, (height, 1, 1))
        return Image.fromarray(gradient.astype(np.uint8)), "gradient"
    color = tuple(random.randint(0, 255) for _ in range(3))
    return Image.new("RGB", (width, height), color), "solid"


def make_paper_background(width, height):
    base = random.randint(222, 250)
    tint = random.randint(0, 10)
    color = (base, max(0, base - tint), max(0, base - tint - random.randint(0, 6)))
    img = Image.new("RGB", (width, height), color)
    if random.random() < 0.3:
        draw = ImageDraw.Draw(img)
        spacing = random.randint(14, 26)
        line_color = tuple(max(0, c - random.randint(15, 30)) for c in color)
        for y in range(0, height, spacing):
            draw.line([(0, y), (width, y)], fill=line_color, width=1)
    return img, "paper"


def choose_background(width, height, style):
    if random.random() < LIGHT_BG_PROB or (style["prefer_paper_bg"] and random.random() < 0.7):
        return make_paper_background(width, height)
    return make_photo_or_solid_background(width, height)

def add_noise(image):
    arr = np.asarray(image, dtype=np.float32)
    sigma = random.uniform(3, 20)
    arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def jpeg_recompress(image):
    quality = random.randint(*JPEG_QUALITY_RANGE)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def random_perspective(image):
    w, h = image.size
    magnitude = random.uniform(0.02, 0.08)
    dx = int(w * magnitude)
    dy = int(h * magnitude)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (random.randint(0, dx), random.randint(0, dy)),
        (w - random.randint(0, dx), random.randint(0, dy)),
        (w - random.randint(0, dx), h - random.randint(0, dy)),
        (random.randint(0, dx), h - random.randint(0, dy)),
    ]
    coeffs = find_perspective_coeffs(src, dst)
    return image.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def find_perspective_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p2[0], p2[1], 1, 0, 0, 0, -p1[0] * p2[0], -p1[0] * p2[1]])
        matrix.append([0, 0, 0, p2[0], p2[1], 1, -p1[1] * p2[0], -p1[1] * p2[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(pa, dtype=np.float64).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()


def motion_blur(image):
    size = random.choice([3, 5, 7])
    kernel = np.zeros((size, size))
    if random.random() < 0.5:
        kernel[size // 2, :] = 1.0
    else:
        kernel[:, size // 2] = 1.0
    kernel /= size
    arr = np.asarray(image, dtype=np.float32)
    from scipy.signal import convolve2d
    blurred = np.stack(
        [convolve2d(arr[..., c], kernel, mode="same", boundary="symm") for c in range(3)],
        axis=-1,
    )
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


def apply_degradations(image):
    if random.random() < PERSPECTIVE_PROB:
        image = random_perspective(image)
    if random.random() < MOTION_BLUR_PROB:
        image = motion_blur(image)
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
    image = add_noise(image)
    if random.random() < JPEG_RECOMPRESS_PROB:
        image = jpeg_recompress(image)
    return image

def create_text_mask(text, font, stroke_width=0):
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    mode = random.choices(list(TEXT_MODE_WEIGHTS.keys()), weights=list(TEXT_MODE_WEIGHTS.values()))[0]

    if mode in ("straight", "diagonal"):
        bbox = dummy_draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        x0, y0, x1, y1 = bbox if bbox else (0, 0, 20, 20)
        w, h = x1 - x0, y1 - y0
        pad = 40 + stroke_width * 2

        img_outer = Image.new("RGBA", (w + pad, h + pad), (0, 0, 0, 0))
        ImageDraw.Draw(img_outer).text(
            (pad // 2 - x0, pad // 2 - y0), text, font=font,
            fill=(255, 255, 255, 255), stroke_width=stroke_width, stroke_fill=(255, 255, 255, 255),
        )

        img_inner = Image.new("RGBA", (w + pad, h + pad), (0, 0, 0, 0))
        ImageDraw.Draw(img_inner).text(
            (pad // 2 - x0, pad // 2 - y0), text, font=font, fill=(255, 255, 255, 255),
        )

        if mode == "diagonal":
            angle = random.uniform(*DIAGONAL_ANGLE_RANGE)
            img_outer = img_outer.rotate(angle, expand=True, resample=Image.BICUBIC)
            img_inner = img_inner.rotate(angle, expand=True, resample=Image.BICUBIC)

        bbox_outer = img_outer.getbbox()
        if not bbox_outer:
            raise ValueError("пустая маска")
        return img_outer.crop(bbox_outer), img_inner.crop(bbox_outer)

    chars = list(text)
    char_widths = [dummy_draw.textlength(c, font=font) for c in chars]
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    max_h = bbox[3] - bbox[1] if bbox else 20
    total_w = sum(char_widths) or 1
    arch_height = total_w * random.uniform(0.08, 0.18)
    a = arch_height / ((total_w / 2) ** 2 + 1e-6)
    direction = random.choice([1, -1])
    img_w = int(total_w + max_h * 2)
    img_h = int(max_h + arch_height + max_h * 2)
    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    current_x = max_h
    for i, c in enumerate(chars):
        c_w = char_widths[i]
        if not c.strip():
            current_x += c_w
            continue
        rel_x = (current_x + c_w / 2) - (img_w / 2)
        y_offset = a * (rel_x ** 2)
        y_pos = y_offset + max_h if direction == 1 else arch_height - y_offset + max_h
        slope = 2 * a * rel_x * direction
        angle_deg = -np.degrees(np.arctan(slope))
        c_bbox = dummy_draw.textbbox((0, 0), c, font=font)
        cx0, cy0, cx1, cy1 = c_bbox if c_bbox else (0, 0, int(c_w), int(max_h))
        cw, ch = cx1 - cx0, cy1 - cy0
        c_img = Image.new("RGBA", (int(cw) + 20, int(ch) + 20), (0, 0, 0, 0))
        ImageDraw.Draw(c_img).text((10 - cx0, 10 - cy0), c, font=font, fill=(255, 255, 255, 255))
        c_img = c_img.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
        img.paste(c_img, (int(current_x), int(y_pos)), c_img)
        current_x += c_w
    bbox = img.getbbox()
    if not bbox:
        raise ValueError("пустая маска")
    cropped = img.crop(bbox)
    return cropped, cropped


def generate_image(text, domain, by_category):
    style = style_params_for_domain(domain)
    font_path, font_category = pick_font(by_category, text, forced_category=style["font_category"])

    font_size = random.randint(28, 60) if font_category == "comic" else random.randint(20, 50)
    font = ImageFont.truetype(font_path, font_size)

    stroke_width = random.randint(1, 3) if style["force_bold_stroke"] else 0
    mask_outer, mask_inner = create_text_mask(text, font, stroke_width=stroke_width)
    tw, th = mask_outer.size

    padding_x = random.randint(10, 30)
    padding_y = random.randint(8, 20)
    width, height = tw + 2 * padding_x, th + 2 * padding_y

    bg_img, bg_group = choose_background(width, height, style)
    bg = bg_img.convert("RGBA")

    if style["bright_colors"]:
        text_color = random.choice([
            (255, 40, 40), (255, 210, 0), (0, 160, 255), (255, 255, 255), (20, 20, 20),
        ])
    else:
        text_color = tuple(random.randint(0, 255) for _ in range(3))

    if stroke_width > 0:
        outline_color = (0, 0, 0) if sum(text_color) > 380 else (255, 255, 255)
        outline_layer = Image.new("RGBA", mask_outer.size, outline_color + (255,))
        outline_text = Image.composite(outline_layer, Image.new("RGBA", mask_outer.size, (0, 0, 0, 0)), mask_outer)
        fill_layer = Image.new("RGBA", mask_inner.size, text_color + (255,))
        fill_text = Image.composite(fill_layer, Image.new("RGBA", mask_inner.size, (0, 0, 0, 0)), mask_inner)
        colored_text = Image.alpha_composite(outline_text, fill_text)
    else:
        color_layer = Image.new("RGBA", mask_outer.size, text_color + (255,))
        colored_text = Image.composite(color_layer, Image.new("RGBA", mask_outer.size, (0, 0, 0, 0)), mask_outer)

    if style["draw_box"]:
        box_color = tuple(random.randint(0, 255) for _ in range(3))
        box_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(box_layer).rectangle(
            [2, 2, width - 3, height - 3], outline=box_color + (255,), width=random.randint(2, 4)
        )
        bg = Image.alpha_composite(bg, box_layer)

    bg.paste(colored_text, (padding_x, padding_y), colored_text)
    image = bg.convert("RGB")

    meta = {"bg_group": bg_group, "font_category": font_category}
    return apply_degradations(image), meta


def generate_synthetic():
    set_seed(SEED)
    SYNTH_IMAGES.mkdir(parents=True, exist_ok=True)
    by_category = get_fonts()
    manifest_rows = []
    total = N_SYNTHETIC_RU + N_SYNTHETIC_EN
    pbar = tqdm(total=total)

    i = 0
    while i < N_SYNTHETIC_RU:
        text, domain = random_ru_text()
        try:
            image, meta = generate_image(text, domain, by_category)
        except Exception:
            continue
        filename = f"synth_ru_{i:06d}.png"
        image.save(SYNTH_IMAGES / filename, quality=95)
        source_group = meta["bg_group"] if meta["bg_group"].startswith("bg_") else filename
        manifest_rows.append(
            (filename, source_group, meta["bg_group"], 0, text, domain, meta["font_category"], "ru")
        )
        i += 1
        pbar.update(1)

    i = 0
    while i < N_SYNTHETIC_EN:
        text, domain = random_en_text()
        try:
            image, meta = generate_image(text, domain, by_category)
        except Exception:
            continue
        filename = f"synth_en_{i:06d}.png"
        image.save(SYNTH_IMAGES / filename, quality=95)
        source_group = meta["bg_group"] if meta["bg_group"].startswith("bg_") else filename
        manifest_rows.append(
            (filename, source_group, meta["bg_group"], 0, text, domain, meta["font_category"], "en")
        )
        i += 1
        pbar.update(1)

    pbar.close()

    with open(SYNTH_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "source_group", "background_group", "base_label",
            "text", "domain", "font_category", "lang",
        ])
        writer.writerows(manifest_rows)

    print(f"{N_SYNTHETIC_RU} RU + {N_SYNTHETIC_EN} EN картинок")

if __name__ == "__main__":
    download_backgrounds()
    generate_synthetic()
