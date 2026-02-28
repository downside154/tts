"""Celery task definitions.

Defines the voice cloning processing task that orchestrates
the full analysis pipeline as a background job.
"""

import json
import logging
import traceback
from pathlib import Path

from celery import Celery

from app.config import settings
from app.errors import PipelineError
from app.models.db import Job, JobStatus, SpeakerProfile, get_session_factory
from app.pipelines.analyze import build_speaker_profile

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

# Pipeline errors that should not be retried (user-actionable)
_NON_RETRYABLE_ERRORS = frozenset({
    "no_speech",
    "insufficient_audio",
    "audio_corrupt",
})


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
        job.stage = "analyzing"
        job.progress = 0.0
        db.commit()

        input_path = Path(job.input_file_path)  # type: ignore[arg-type]
        profile_data = build_speaker_profile(input_path, job_id)

        speaker_profile = SpeakerProfile(
            id=profile_data["id"],
            name=input_path.stem,
            embedding_path=profile_data["embedding_path"],
            segments_json=json.dumps(profile_data["segments"]),
            metadata_json=json.dumps({
                "source_file": profile_data["source_file"],
                "speaker_count": profile_data["speaker_count"],
                "dominant_speaker_id": profile_data["dominant_speaker_id"],
                "quality_summary": profile_data["quality_summary"],
                "total_duration_s": profile_data["total_duration_s"],
            }),
        )
        db.add(speaker_profile)

        job.speaker_profile_id = speaker_profile.id
        job.status = JobStatus.COMPLETED
        job.stage = "done"
        job.progress = 100.0
        db.commit()
        logger.info("Job %s completed", job_id)

    except Exception as exc:
        db.rollback()

        # Extract error_code and user_message from typed pipeline errors
        if isinstance(exc, PipelineError):
            error_code = exc.error_code
            error_message = exc.user_message
        else:
            error_code = "processing_error"
            error_message = f"{type(exc).__name__}: {exc}"

        # Update job as failed
        try:
            job = db.get(Job, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error_code = error_code
                job.error_message = error_message
                db.commit()
        except Exception:
            logger.error("Failed to update job %s status: %s", job_id, traceback.format_exc())

        logger.error("Job %s failed (%s): %s", job_id, error_code, exc)

        # Only retry errors that are not user-actionable
        if error_code not in _NON_RETRYABLE_ERRORS and self.request.retries < self.max_retries:
            raise self.retry(exc=exc) from exc
    finally:
        db.close()
