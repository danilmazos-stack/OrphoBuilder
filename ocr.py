import re

import pytesseract
from PIL import Image, ImageOps, ImageStat

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
LANGS = "rus+eng"
PSM = 3  # 3 = авто (абзацы), 6 = один сплошной блок текста

# Отключаем словари Tesseract, чтобы он не «исправлял» ошибки сам
# и читал текст так, как он написан. Если распознавание ухудшится, уберите эту строку.
CONFIG = f"--oem 1 --psm {PSM} -c load_system_dawg=0 -c load_freq_dawg=0"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class OcrError(Exception):
    pass


def _prepare(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("L")
    if ImageStat.Stat(img).mean[0] < 110:      # тёмный фон: инвертируем
        img = ImageOps.invert(img)
    w, h = img.size
    if w < 1500:                               # маленькие картинки увеличиваем
        k = 2 if w >= 750 else 3
        img = img.resize((w * k, h * k), Image.Resampling.LANCZOS)
    return ImageOps.autocontrast(img)


def read_text(path: str) -> str:
    try:
        img = _prepare(Image.open(path))
        data = pytesseract.image_to_data(
            img, lang=LANGS, config=CONFIG, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractNotFoundError:
        raise OcrError(
            "Tesseract не найден. Проверьте путь TESSERACT_CMD в ocr.py "
            "или установите программу."
        )
    except pytesseract.TesseractError as e:
        raise OcrError(f"Ошибка Tesseract: {e}")
    except OSError as e:
        raise OcrError(f"Не удалось открыть картинку: {e}")

    # склеиваем строки внутри абзаца, абзацы разделяем переводом строки
    paragraphs: dict[tuple[int, int], list[str]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i])
        paragraphs.setdefault(key, []).append(word)
    return _clean("\n".join(" ".join(words) for words in paragraphs.values()))

def _clean(text: str) -> str:
    text = re.sub(r"\s*[|¦\\_~]+\s*", " ", text)        # символы от рамок и линеек
    text = re.sub(r"(?<=\s)[,;:]+(?=\w)", "", text)      # запятая перед словом
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)         # пробел перед знаком
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()