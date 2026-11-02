import re
from itertools import islice
from pathlib import Path

import requests
from spylls.hunspell import Dictionary

DICT_DIR = Path(__file__).parent / "dicts"
BASE_URL = "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries"
LANGS = ("ru", "en")


def _download(lang: str) -> None:
    """Скачивает словарь, если его ещё нет в папке dicts."""
    DICT_DIR.mkdir(exist_ok=True)
    for ext in ("aff", "dic"):
        target = DICT_DIR / f"{lang}.{ext}"
        if target.exists():
            continue
        r = requests.get(f"{BASE_URL}/{lang}/index.{ext}", timeout=60)
        r.raise_for_status()
        target.write_bytes(r.content)


def _load(lang: str) -> Dictionary:
    _download(lang)
    return Dictionary.from_files(str(DICT_DIR / lang))


# загружается один раз при старте программы (русский словарь грузится несколько секунд)
_DICTS = {lang: _load(lang) for lang in LANGS}

CUSTOM_FILE = DICT_DIR / "custom.txt"

def _load_custom() -> set[str]:
    if not CUSTOM_FILE.exists():
        return set()
    lines = CUSTOM_FILE.read_text(encoding="utf-8").splitlines()
    return {ln.strip().lower() for ln in lines if ln.strip()}


_CUSTOM = _load_custom()


def _lang_of(word: str) -> str:
    return "ru" if re.search(r"[а-яё]", word.lower()) else "en"

ALPHABET = {
    "ru": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "en": "abcdefghijklmnopqrstuvwxyz",
}


def _edits1(word: str, alphabet: str) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    swaps = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in alphabet]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + swaps + replaces + inserts)

def is_known(word: str) -> bool:
    """Есть ли слово в словаре. Цифры и знаки считаются «известными»."""
    w = word.strip("-")
    if w.lower() in _CUSTOM:
        return True
    if not re.search(r"[A-Za-zА-Яа-яЁё]", w):
        return True
    return _DICTS[_lang_of(w)].lookup(w)


def suggest(word: str, limit: int = 8) -> list[str]:
    """Варианты исправления: слова на одну правку от исходного + подсказки Hunspell."""
    w = word.strip("-")
    if not re.search(r"[A-Za-zА-Яа-яЁё]", w):
        return []
    lang = _lang_of(w)
    d = _DICTS[lang]
    low = w.lower()

    near = sorted(c for c in _edits1(low, ALPHABET[lang]) if c != low and (d.lookup(c) or c in _CUSTOM))
    if w[0].isupper():
        near = [c.capitalize() for c in near]

    result = near[:limit]
    for s in islice(d.suggest(w), limit):
        if s not in result:
            result.append(s)
    return result