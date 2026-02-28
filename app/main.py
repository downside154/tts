"""FastAPI application entry point.

Creates and configures the FastAPI app instance, includes routers,
sets up CORS middleware, and defines lifespan events.
"""

from fastapi import FastAPI

app = FastAPI(title="Korean Voice Cloning TTS", version="0.1.0")
