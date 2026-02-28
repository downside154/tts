"""FastAPI application entry point.

Creates and configures the FastAPI app instance, includes routers,
sets up CORS middleware, and defines lifespan events.
"""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Korean Voice Cloning TTS", version="0.1.0")

app.include_router(router, prefix="/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}
