from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class History(Base):
    __tablename__ = "history"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    mode: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(20))      # text | file
    model: Mapped[str] = mapped_column(String(100))
    original: Mapped[str] = mapped_column(Text)
    result: Mapped[str] = mapped_column(Text)
    duration_ms: Mapped[int] = mapped_column(Integer)