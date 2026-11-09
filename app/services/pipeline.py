from app.config import MODEL
from app.services import llm


def run(text: str, model: str = MODEL) -> dict:
    return llm.analyze(text, model)