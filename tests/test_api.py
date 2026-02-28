"""Tests for the FastAPI API endpoints.

Covers voice upload, job status retrieval, synthesis requests,
audio retrieval, and error handling.
"""

import io
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.models.db import Base, Job, SpeakerProfile, get_db

_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(bind=_engine)


def _override_get_db() -> Iterator[Session]:
    db = _TestSession()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def _noop_lifespan(app: FastAPI):
    yield


@pytest.fixture(autouse=True)
def _setup_db():
    Base.metadata.create_all(_engine)
    yield
    Base.metadata.drop_all(_engine)


@pytest.fixture
def client():
    from app.main import app

    app.dependency_overrides[get_db] = _override_get_db
    app.router.lifespan_context = _noop_lifespan
    with patch("app.workers.tasks.process_voice_clone") as mock_task:
        mock_task.delay = lambda *a, **kw: None
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


def test_upload_wav_success(client):
    wav_data = b"\x00" * 1024
    response = client.post(
        "/v1/voices/clone",
        files={"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"

    # Verify job exists in DB
    db = _TestSession()
    job = db.get(Job, data["job_id"])
    assert job is not None
    assert job.status.value == "pending"
    db.close()


def test_upload_invalid_type_rejected(client):
    response = client.post(
        "/v1/voices/clone",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 422
    assert "Invalid file type" in response.json()["detail"]


def test_get_job_status(client):
    response = client.post(
        "/v1/voices/clone",
        files={"file": ("test.mp3", io.BytesIO(b"\x00" * 512), "audio/mpeg")},
    )
    job_id = response.json()["job_id"]

    response = client.get(f"/v1/jobs/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "pending"


def test_get_job_not_found(client):
    response = client.get("/v1/jobs/nonexistent-id")
    assert response.status_code == 404


def test_synthesize_returns_501(client):
    response = client.post(
        "/v1/tts/synthesize",
        json={"speaker_profile_id": "abc", "text": "안녕하세요"},
    )
    assert response.status_code == 501


def test_get_audio_returns_501(client):
    response = client.get("/v1/audio/some-id")
    assert response.status_code == 501


def test_list_voices_empty(client):
    response = client.get("/v1/voices")
    assert response.status_code == 200
    assert response.json() == []


def test_delete_voice_success(client):
    """Deleting an existing speaker profile returns success."""
    profile_id = str(uuid.uuid4())
    db = _TestSession()
    profile = SpeakerProfile(id=profile_id, name="test-speaker")
    db.add(profile)
    db.commit()
    db.close()

    response = client.delete(f"/v1/voices/{profile_id}")
    assert response.status_code == 200
    assert profile_id in response.json()["detail"]

    # Verify it's actually deleted
    db = _TestSession()
    assert db.get(SpeakerProfile, profile_id) is None
    db.close()


def test_delete_voice_not_found(client):
    """Deleting a nonexistent speaker profile returns 404."""
    response = client.delete("/v1/voices/nonexistent-id")
    assert response.status_code == 404
    assert "Speaker profile not found" in response.json()["detail"]


def test_health_endpoint(client):
    """Health endpoint returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_voices_with_profiles(client):
    """Listing voices returns existing speaker profiles."""
    db = _TestSession()
    profile = SpeakerProfile(id=str(uuid.uuid4()), name="speaker-1")
    db.add(profile)
    db.commit()
    db.close()

    response = client.get("/v1/voices")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "speaker-1"
