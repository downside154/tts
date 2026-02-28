"""API route definitions for voice cloning and TTS endpoints.

Includes endpoints for voice upload, job status, speech synthesis,
audio retrieval, and speaker profile management.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.schemas import (
    CloneResponse,
    DeleteResponse,
    JobResponse,
    SpeakerProfileResponse,
    SynthesizeRequest,
)
from app.config import settings
from app.models.db import Job, JobStatus, SpeakerProfile, get_db

router = APIRouter()

ALLOWED_MIME_PREFIXES = ("audio/", "video/")


@router.post("/voices/clone", response_model=CloneResponse)
async def clone_voice(file: UploadFile, db: Session = Depends(get_db)):
    content_type = file.content_type or ""
    if not any(content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{content_type}'. Only audio and video files are accepted.",
        )

    job_id = str(uuid.uuid4())

    storage_dir = Path(settings.storage_path) / job_id
    storage_dir.mkdir(parents=True, exist_ok=True)
    file_path = storage_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        input_file_path=str(file_path),
    )
    db.add(job)
    db.commit()

    return CloneResponse(job_id=job_id, status=JobStatus.PENDING.value)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(
        job_id=job.id,
        status=job.status.value,
        stage=job.stage,
        progress=job.progress,
        error_message=job.error_message,
        speaker_profile_id=job.speaker_profile_id,
    )


@router.post("/tts/synthesize", status_code=501)
async def synthesize(request: SynthesizeRequest):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/audio/{audio_id}", status_code=501)
async def get_audio(audio_id: str):
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get("/voices", response_model=list[SpeakerProfileResponse])
async def list_voices(db: Session = Depends(get_db)):
    profiles = db.query(SpeakerProfile).all()
    return [
        SpeakerProfileResponse(id=p.id, name=p.name, created_at=p.created_at)
        for p in profiles
    ]


@router.delete("/voices/{voice_id}", response_model=DeleteResponse)
async def delete_voice(voice_id: str, db: Session = Depends(get_db)):
    profile = db.get(SpeakerProfile, voice_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Speaker profile not found")
    db.delete(profile)
    db.commit()
    return DeleteResponse(detail=f"Speaker profile '{voice_id}' deleted")
