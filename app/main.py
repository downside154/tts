"""FastAPI application entry point.

Creates and configures the FastAPI app instance, includes routers,
sets up CORS middleware, and defines lifespan events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import create_engine

from app.api.routes import router
from app.config import settings
from app.models.db import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    engine.dispose()
    yield


app = FastAPI(title="Korean Voice Cloning TTS", version="0.1.0", lifespan=lifespan)

app.include_router(router, prefix="/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}
