"""Tests for Celery worker tasks.

Covers the process_voice_clone task including success, failure,
retry logic, and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import sessionmaker

from app.models.db import Base, Job, JobStatus, SpeakerProfile

_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(bind=_engine)


@pytest.fixture(autouse=True)
def _setup_db():
    Base.metadata.create_all(_engine)
    yield
    Base.metadata.drop_all(_engine)


def _create_job(job_id: str = "test-job-1") -> Job:
    db = _TestSession()
    job = Job(id=job_id, status=JobStatus.PENDING, input_file_path="/tmp/test.wav")
    db.add(job)
    db.commit()
    db.close()
    return job


def _make_failing_session_class(fail_on_commit: int, error_cls=RuntimeError, message="Simulated error"):
    """Create a session class that fails on the Nth commit call."""
    original = _TestSession

    class FailingSession:
        def __init__(self):
            self._real = original()
            self._commit_count = 0

        def __getattr__(self, name):
            return getattr(self._real, name)

        def get(self, *args, **kwargs):
            return self._real.get(*args, **kwargs)

        def commit(self):
            self._commit_count += 1
            if self._commit_count == fail_on_commit:
                raise error_cls(message)
            self._real.commit()

        def rollback(self):
            self._real.rollback()

        def close(self):
            self._real.close()

    return FailingSession


def _make_always_fail_session_class():
    """Create a session class where commit always fails."""
    original = _TestSession

    class AlwaysFailSession:
        def __init__(self):
            self._real = original()

        def __getattr__(self, name):
            return getattr(self._real, name)

        def get(self, *args, **kwargs):
            return self._real.get(*args, **kwargs)

        def commit(self):
            raise RuntimeError("Commit always fails")

        def rollback(self):
            self._real.rollback()

        def close(self):
            self._real.close()

    return AlwaysFailSession


_MOCK_PROFILE = {
    "id": "test-profile-id",
    "created_at": "2024-01-01T00:00:00+00:00",
    "source_file": "test.wav",
    "embedding_path": "/tmp/embedding.npy",
    "segments": [{"path": "/tmp/seg.wav", "start": 0.0, "end": 5.0, "quality_score": 0.9}],
    "total_duration_s": 15.0,
    "speaker_count": 1,
    "dominant_speaker_id": "SPEAKER_00",
    "quality_summary": {"mean_snr": 25.0, "clipped_segments": 0},
}


@pytest.fixture(autouse=True)
def _mock_profile_builder():
    with patch("app.workers.tasks.build_speaker_profile", return_value=_MOCK_PROFILE):
        yield


@pytest.fixture
def _mock_factory():
    with patch("app.workers.tasks.get_session_factory", return_value=_TestSession):
        yield


class TestProcessVoiceClone:
    def test_successful_completion(self, _mock_factory) -> None:
        """Job transitions from pending to completed with speaker profile."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-success")

        process_voice_clone.run("job-success")

        db = _TestSession()
        job = db.get(Job, "job-success")
        assert job.status == JobStatus.COMPLETED
        assert job.stage == "done"
        assert job.progress == 100.0
        assert job.speaker_profile_id == "test-profile-id"

        profile = db.get(SpeakerProfile, "test-profile-id")
        assert profile is not None
        assert profile.name == "test"
        assert profile.embedding_path == "/tmp/embedding.npy"
        db.close()

    def test_job_not_found(self, _mock_factory) -> None:
        """Missing job ID logs error and returns without crash."""
        from app.workers.tasks import process_voice_clone

        process_voice_clone.run("nonexistent-job")

    def test_processing_updates_status(self, _mock_factory) -> None:
        """Job status transitions through processing to completed."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-stage")

        process_voice_clone.run("job-stage")

        db = _TestSession()
        job = db.get(Job, "job-stage")
        assert job.status == JobStatus.COMPLETED
        assert job.stage == "done"
        assert job.progress == 100.0
        db.close()

    def test_failure_records_error(self) -> None:
        """Exception during processing marks job as failed with error message."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-fail")

        failing_cls = _make_failing_session_class(fail_on_commit=2)

        # Use push_request to set up Celery request context
        process_voice_clone.push_request(retries=2)
        try:
            with patch("app.workers.tasks.get_session_factory", return_value=lambda: failing_cls()):
                process_voice_clone.run("job-fail")
        finally:
            process_voice_clone.pop_request()

        db = _TestSession()
        job = db.get(Job, "job-fail")
        assert job.status == JobStatus.FAILED
        assert "RuntimeError" in job.error_message
        db.close()

    def test_failure_with_retry(self) -> None:
        """Exception with retries remaining triggers retry."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-retry")

        failing_cls = _make_failing_session_class(
            fail_on_commit=2, error_cls=ValueError, message="Simulated processing error"
        )

        mock_retry = MagicMock(side_effect=ValueError("retry triggered"))

        process_voice_clone.push_request(retries=0)
        try:
            with (
                patch("app.workers.tasks.get_session_factory", return_value=lambda: failing_cls()),
                patch.object(type(process_voice_clone._get_current_object()), "retry", mock_retry),
                pytest.raises(ValueError, match="retry triggered"),
            ):
                process_voice_clone.run("job-retry")
        finally:
            process_voice_clone.pop_request()

        mock_retry.assert_called_once()

    def test_error_handler_failure_is_swallowed(self) -> None:
        """If error handler itself fails, the error is logged but doesn't propagate."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-double-fail")

        always_fail_cls = _make_always_fail_session_class()

        process_voice_clone.push_request(retries=2)
        try:
            with patch("app.workers.tasks.get_session_factory", return_value=lambda: always_fail_cls()):
                # Should not raise even though all commits fail
                process_voice_clone.run("job-double-fail")
        finally:
            process_voice_clone.pop_request()
