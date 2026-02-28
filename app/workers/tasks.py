"""Celery task definitions.

Defines the voice cloning processing task that orchestrates
the full analysis pipeline as a background job.
"""

import logging
import traceback

from celery import Celery

from app.config import settings
from app.models.db import Job, JobStatus, get_session_factory

logger = logging.getLogger(__name__)

celery = Celery(
    "tasks",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


@celery.task(
    bind=True,
    max_retries=2,
    soft_time_limit=600,
    time_limit=660,
)
def process_voice_clone(self, job_id: str) -> None:
    factory = get_session_factory()
    db = factory()
    try:
        job = db.get(Job, job_id)
        if not job:
            logger.error("Job %s not found", job_id)
            return

        job.status = JobStatus.PROCESSING
        job.stage = "ingest"
        job.progress = 0.0
        db.commit()

        # TODO: call ingest pipeline (Task 1.5)
        # ingest.extract_audio(Path(job.input_file_path))

        job.status = JobStatus.COMPLETED
        job.stage = "done"
        job.progress = 100.0
        db.commit()
        logger.info("Job %s completed", job_id)

    except Exception as exc:
        db.rollback()
        # Update job as failed
        try:
            job = db.get(Job, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error_message = f"{type(exc).__name__}: {exc}"
                db.commit()
        except Exception:
            logger.error("Failed to update job %s status: %s", job_id, traceback.format_exc())
        logger.error("Job %s failed: %s", job_id, exc)
        raise self.retry(exc=exc) if self.request.retries < self.max_retries else None
    finally:
        db.close()
