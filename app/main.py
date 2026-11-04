import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.db import models  # noqa: F401

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        Base.metadata.create_all(engine)
    except Exception as e:
        log.warning("База данных недоступна (%s). История работать не будет.", type(e).__name__)
    yield


app = FastAPI(title="Text Fixer API", lifespan=lifespan)
app.include_router(router)
from app.api.routes import router
from app.db import models  # noqa: F401  (регистрирует таблицы)
from app.db.session import Base, engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Text Fixer API", lifespan=lifespan)
app.include_router(router)