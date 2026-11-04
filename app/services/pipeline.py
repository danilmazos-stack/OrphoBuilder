from app.config import MODEL
from app.services import llm


def run(text: str, show_changes: bool = False, model: str = MODEL) -> str:
    """Берёт последний результат из потокового генератора."""
    last = ""
    for part in llm.fix_text_iter(text, model, show_changes):
        last = part
    return last