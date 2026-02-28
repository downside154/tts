"""Pydantic models for API request and response schemas.

Defines the data shapes for all API contract endpoints including
voice cloning, job status, synthesis, and audio retrieval.
"""

from datetime import datetime

from pydantic import BaseModel


class CloneResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    stage: str | None = None
    progress: float | None = None
    error_message: str | None = None
    speaker_profile_id: str | None = None


class SynthesizeRequest(BaseModel):
    speaker_profile_id: str
    text: str
    language: str = "ko"


class SynthesizeResponse(BaseModel):
    audio_id: str
    download_url: str


class SpeakerProfileResponse(BaseModel):
    id: str
    name: str
    created_at: datetime


class DeleteResponse(BaseModel):
    detail: str
