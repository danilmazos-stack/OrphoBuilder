from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str
    mode: str = "correct"        # correct | stylize (стилизации пока нет)
    show_changes: bool = False


class Change(BaseModel):
    was: str
    now: str
    type: str


class CorrectionResponse(BaseModel):
    original: str
    corrected: str
    changes: list[Change]
    errors_found: int
    duration_ms: int