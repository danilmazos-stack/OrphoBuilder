from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str
    mode: str = "correct"        # correct | stylize (стилизации пока нет)
    show_changes: bool = False


class CorrectionResponse(BaseModel):
    result: str
    duration_ms: int