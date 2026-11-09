import base64
import difflib
import json
import re
from functools import lru_cache
import requests
from app.services import spell
from app.services.ocr import read_text, OcrError
from app.services.extract import read_any

from app.config import OLLAMA_URL as OLLAMA_BASE, MODEL

OLLAMA_URL = OLLAMA_BASE + "/api/chat"
TAGS_URL = OLLAMA_BASE + "/api/tags"  # впишите свою модель из ollama list
TIMEOUT = 300
USE_YANDEX = True

SKIP_CLEAN = True  # не отправлять модели предложения без подозрительных слов
PROTECT_KNOWN_WORDS = False  # РАЗРЕШИТЬ модели менять слова, даже если они есть в словаре (важно для "слава" -> "слова")
SHOW_RECOGNIZED = False  # True: показывать текст, прочитанный с картинки

SYSTEM_PROMPT = (
    "You are a proofreader for Russian and English text. "
    "Fix ONLY spelling, grammar, capitalization and punctuation errors in the given text "
    'and return it as JSON: {"corrected": "..."}.\n'
    "RULES:\n"
    "1. NEVER translate. Russian stays Russian, English stays English.\n"
    "2. You MAY change a word if it is spelled correctly but makes NO sense in the context "
    "(e.g., 'слава' instead of 'слова', 'луг' instead of 'лук'). Choose the word that fits the meaning.\n"
    "3. NEVER delete, add or reorder words unnecessarily. Keep the original structure.\n"
    "4. Capitalize names and the first word of a sentence.\n"
    "5. Leave everything that is already correct exactly as it is, including digits and style.\n"
    "6. Output ONLY the JSON, no comments or explanations."
)

SCHEMA = {
    "type": "object",
    "properties": {"corrected": {"type": "string"}},
    "required": ["corrected"],
}

EXAMPLES = [
    ("Я вчира пошол в магазин и купил хлеп.", "Я вчера пошёл в магазин и купил хлеб."),
    ("Their going to the store tomorow.", "They're going to the store tomorrow."),
    ("Иля и Витя лубят играть с сабакой.", "Илья и Витя любят играть с собакой."),
    ("мы живем в москве", "Мы живём в Москве"),
    ("ты умеешь исправлять слава", "Ты умеешь исправлять слова"),
]

PROMPT_OCR = (
    "Transcribe all text from the image exactly as written, including any mistakes. "
    "Do not correct, translate or comment. Output only the text."
)

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")


class OllamaError(Exception):
    pass


# ---------- словарь (с кешем, чтобы не считать одно и то же много раз) ----------

def _norm(w: str) -> str:
    return re.sub(r"[^\w]", "", w).lower().replace("ё", "е")


def _lang(w: str) -> str:
    return "ru" if re.search(r"[а-я]", w) else "en"


@lru_cache(maxsize=20000)
def _is_known(word: str) -> bool:
    return spell.is_known(word)


@lru_cache(maxsize=5000)
def _candidates(word: str) -> frozenset:
    return frozenset(spell.suggest(word))


@lru_cache(maxsize=2000)
def _hints(text: str) -> str:
    lines, seen = [], set()
    for w in WORD_RE.findall(text):
        if len(w) <= 2 or w.lower() in seen:
            continue
        seen.add(w.lower())
        cands = sorted(
            _candidates(w),
            key=lambda c: difflib.SequenceMatcher(None, _norm(w), _norm(c)).ratio(),
            reverse=True,
        )[:6]
        if cands:
            lines.append(f"{w}: {', '.join(cands)}")
    return "\n".join(lines)


def _build_user(text: str) -> str:
    # Просто отправляем текст модели, без лишних подсказок, чтобы не путать маленькую модель
    return text


def _pick(wa: str, wb: str) -> str:
    na, nb = _norm(wa), _norm(wb)
    if na == nb:
        return wb
    if PROTECT_KNOWN_WORDS and _is_known(wa) and not _is_known(wb):
        return wa
    return wb


def _guard(original: str, corrected: str) -> str:
    a, b = original.split(), corrected.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("equal", "delete"):
            out.extend(a[i1:i2])
        elif op == "insert":
            continue
        elif (i2 - i1) == (j2 - j1):
            for wa, wb in zip(a[i1:i2], b[j1:j2]):
                out.append(_pick(wa, wb))
        else:
            out.extend(b[j1:j2])
    return " ".join(out)


# ---------- обмен с Ollama ----------

def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _post(messages: list[dict], model: str, schema: dict | None = None) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": "5m", # Выгружать модель из памяти через 5 минут, чтобы освободить RAM/CPU
        "options": {
            "temperature": 0,       # 0 требует меньше вычислений, чем 0.5
            "num_ctx": 512,         # БЫЛО 4096. Для исправления предложений 512 более чем достаточно!
            "num_predict": 150,     # БЫЛО 500. Ограничиваем максимальную длину ответа
            "top_k": 20,            # Дополнительные ограничения для ускорения CPU
            "top_p": 0.9,
        },
        "messages": messages,
    }
    if schema:
        payload["format"] = schema
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise OllamaError("Ошибка: Ollama не запущена. Запустите её и повторите.")
    except requests.exceptions.RequestException as e:
        raise OllamaError(f"Ошибка запроса к Ollama: {e}")
    return r.json()["message"]["content"]


def _correct_chunk(text: str, model: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for src, fixed in EXAMPLES:
        messages.append({"role": "user", "content": src})
        messages.append({
            "role": "assistant",
            "content": json.dumps({"corrected": fixed}, ensure_ascii=False),
        })
    messages.append({"role": "user", "content": text})

    raw = _post(messages, model, SCHEMA)
    try:
        corrected = json.loads(raw)["corrected"]
    except (json.JSONDecodeError, KeyError):
        return text

    # Возвращаем ответ модели напрямую, без жёсткой фильтрации через _guard
    return corrected


def _needs_model(chunk: str) -> bool:
    s = chunk.strip()
    if not s:
        return False
    if not SKIP_CLEAN:
        return True
    if s[0].islower():
        return True
    return any(not _is_known(w) for w in WORD_RE.findall(s) if len(w) > 2)


# ---------- формат результата ----------

def _label(was: str, now: str) -> str:
    if not was:
        return "добавлено слово"
    if not now:
        return "удалено слово"
    if was.lower() == now.lower():
        return "регистр букв"
    strip = lambda s: re.sub(r"[^\w\s]", "", s)
    if strip(was) == strip(now):
        return "пунктуация"
    return "орфография / грамматика"


def _diff(original: str, corrected: str) -> list[tuple[str, str]]:
    a, b = original.split(), corrected.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [
        (" ".join(a[i1:i2]), " ".join(b[j1:j2]))
        for op, i1, i2, j1, j2 in sm.get_opcodes()
        if op != "equal"
    ]


def _format(original: str, corrected: str, show_changes: bool) -> str:
    if not show_changes:
        return corrected
    lines = [corrected, "", "Правки:"]
    changes = _diff(original, corrected)
    if changes:
        for was, now in changes:
            lines.append(f"• {was or '—'} → {now or '—'} ({_label(was, now)})")
    else:
        lines.append("Ошибок не найдено.")
    return "\n".join(lines)


# ---------- то, что вызывает интерфейс (генераторы: результат идёт по частям) ----------

def correct_iter(text: str, model: str = MODEL):
    """Отдаёт исправленный текст по мере готовности, без оформления."""
    if USE_YANDEX:
        try:
            text = spell.yandex_fix(text)
        except requests.exceptions.RequestException:
            raise OllamaError("Яндекс.Спеллер недоступен. Проверьте интернет.")
        yield text

    parts = re.split(r"(\n+|(?<=[.!?…])[ \t]+)", text)
    result = list(parts)
    for i, part in enumerate(parts):
        if i % 2 or not part.strip():
            continue
        piece = part
        if _needs_model(piece):
            piece = _correct_chunk(piece, model)
        result[i] = piece
        yield "".join(result)


def fix_text_iter(text: str, model: str = MODEL, show_changes: bool = False):
    corrected = text
    try:
        for corrected in correct_iter(text, model):
            yield corrected
    except OllamaError as e:
        yield str(e)
        return
    yield _format(text, corrected, show_changes)


def analyze(text: str, model: str = MODEL) -> dict:
    """Полный результат в виде словаря (для JSON)."""
    corrected = text
    for corrected in correct_iter(text, model):
        pass
    changes = [{"was": w, "now": n, "type": _label(w, n)} for w, n in _diff(text, corrected)]
    return {
        "original": text,
        "corrected": corrected,
        "changes": changes,
        "errors_found": len(changes),
    }


def fix_image_iter(image_path: str, model: str = MODEL, show_changes: bool = False):
    yield "Распознаю текст на картинке..."
    try:
        extracted = read_text(image_path).strip()
    except OcrError as e:
        yield str(e)
        return
    if not extracted:
        yield "Текст на картинке не найден."
        return
    prefix = f"Распознанный текст:\n{extracted}\n\n" if SHOW_RECOGNIZED else ""
    for out in fix_text_iter(extracted, model, show_changes):
        yield prefix + out


def list_models() -> list[str]:
    try:
        r = requests.get(TAGS_URL, timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json()["models"]]
    except requests.exceptions.RequestException:
        return []




def fix_file_iter(path: str, model: str = MODEL, show_changes: bool = False):
    yield "Читаю файл..."
    try:
        text = read_any(path).strip()
    except Exception as e:
        yield f"Не удалось прочитать файл: {e}"
        return
    if not text:
        yield "В файле не найден текст."
        return
    yield from fix_text_iter(text, model, show_changes)