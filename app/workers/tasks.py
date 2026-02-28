"""Celery task definitions.

Defines the voice cloning processing task that orchestrates
the full analysis pipeline as a background job.
"""

from celery import Celery

from app.config import settings

celery = Celery(
    "tasks",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
