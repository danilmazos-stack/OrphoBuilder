import re
from pathlib import Path

import docx   # это python-docx
import pymupdf # это и есть PyMuPDF (MuPDF)
from PIL import Image

from ocr import read_text, OcrError


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _read_docx(path: str) -> str:
    d = docx.Document(path)
    lines = [p.text for p in d.paragraphs]
    for table in d.tables:                      # таблицы добавляются в конец
        for row in table.rows:
            lines.append(" ".join(c.text for c in row.cells))
    return "\n".join(lines)


def _unwrap(text: str) -> str:
    # склеиваем строки одного абзаца, чтобы предложения не рвались на части
    return re.sub(r"(?<![\n.!?:])\n(?!\n)", " ", text)


def _read_pdf(path: str) -> str:
    pages = []
    with pymupdf.open(path) as pdf:
        for page in pdf:
            text = page.get_text().strip()
            if len(text) < 20:                  # скан без текстового слоя: читаем через Tesseract
                pix = page.get_pixmap(dpi=300, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = read_text(img)
            pages.append(text)
    return _unwrap("\n\n".join(pages))


def read_any(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return _read_docx(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext in IMAGE_EXT:
        return read_text(path)
    if ext == ".txt":
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    raise OcrError(f"Формат {ext} не поддерживается (нужен .docx, .pdf, .txt или картинка).")