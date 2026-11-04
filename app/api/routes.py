import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from app.config import MAX_FILE_MB, MODEL
from app.db.models import History
from app.db.session import SessionLocal
from app.schemas import CorrectionResponse, TextRequest
from app.services import extract, llm, pipeline

router = APIRouter(prefix="/api")
ALLOWED = {".docx", ".pdf", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _save(db: Session, source: str, original: str, result: str, started: float) -> int:
    ms = int((time.time() - started) * 1000)
    try:
        db.add(History(mode="correct", source=source, model=MODEL,
                       original=original, result=result, duration_ms=ms))
        db.commit()
    except Exception:
        db.rollback()
    return ms


@router.post("/correct", response_model=CorrectionResponse)
def correct(req: TextRequest, db: Session = Depends(get_db)):
    if not req.text.strip():
        raise HTTPException(400, "Пустой текст")
    if req.mode != "correct":
        raise HTTPException(501, "Этот режим пока не реализован")
    started = time.time()
    result = pipeline.run(req.text, req.show_changes)
    return CorrectionResponse(result=result, duration_ms=_save(db, "text", req.text, result, started))


@router.post("/correct/file", response_model=CorrectionResponse)
def correct_file(file: UploadFile = File(...), show_changes: bool = False,
                 db: Session = Depends(get_db)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, f"Формат {ext or '?'} не поддерживается")
    data = file.file.read()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"Файл больше {MAX_FILE_MB} МБ")

    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{ext}"
        path.write_bytes(data)
        try:
            original = extract.read_any(str(path)).strip()
        except Exception as e:
            raise HTTPException(422, f"Не удалось прочитать файл: {e}")
    if not original:
        raise HTTPException(422, "В файле не найден текст")

    result = pipeline.run(original, show_changes)
    return CorrectionResponse(result=result, duration_ms=_save(db, "file", original, result, started))


@router.get("/history")
def history(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.query(History).order_by(History.id.desc()).limit(limit).all()
    return [{"id": r.id, "created_at": r.created_at, "source": r.source,
             "duration_ms": r.duration_ms} for r in rows]


@router.get("/history/{item_id}")
def history_item(item_id: int, db: Session = Depends(get_db)):
    r = db.get(History, item_id)
    if not r:
        raise HTTPException(404, "Не найдено")
    return {"id": r.id, "original": r.original, "result": r.result, "model": r.model}


@router.get("/models")
def models():
    return llm.list_models()


@router.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(sql("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"ollama": bool(llm.list_models()), "db": db_ok}