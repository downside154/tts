"""Tests for Celery worker tasks.

Covers the process_voice_clone task including success, failure,
retry logic, error categorization, and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import StaticPool, create_engine
from sqlalchemy.orm import sessionmaker

from app.errors import (
    AudioCorruptError,
    InsufficientAudioError,
    NoSpeechDetectedError,
    ProcessingError,
)
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
        assert job is not None
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
        assert job is not None
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
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error_message is not None
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

    def test_no_speech_error_sets_error_code(self, _mock_factory) -> None:
        """NoSpeechDetectedError sets error_code='no_speech' on the job."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-nospeech")

        process_voice_clone.push_request(retries=0)
        try:
            with patch(
                "app.workers.tasks.build_speaker_profile",
                side_effect=NoSpeechDetectedError(stage="diarization"),
            ):
                process_voice_clone.run("job-nospeech")
        finally:
            process_voice_clone.pop_request()

        db = _TestSession()
        job = db.get(Job, "job-nospeech")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error_code == "no_speech"
        assert job.error_message == "No speech detected in the uploaded file"
        db.close()

    def test_insufficient_audio_error_sets_error_code(self, _mock_factory) -> None:
        """InsufficientAudioError sets error_code='insufficient_audio'."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-short")

        process_voice_clone.push_request(retries=0)
        try:
            with patch(
                "app.workers.tasks.build_speaker_profile",
                side_effect=InsufficientAudioError(stage="segment_selection"),
            ):
                process_voice_clone.run("job-short")
        finally:
            process_voice_clone.pop_request()

        db = _TestSession()
        job = db.get(Job, "job-short")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error_code == "insufficient_audio"
        assert job.error_message == "At least 3 seconds of speech required"
        db.close()

    def test_audio_corrupt_error_sets_error_code(self, _mock_factory) -> None:
        """AudioCorruptError sets error_code='audio_corrupt'."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-corrupt")

        process_voice_clone.push_request(retries=0)
        try:
            with patch(
                "app.workers.tasks.build_speaker_profile",
                side_effect=AudioCorruptError(stage="input_validation"),
            ):
                process_voice_clone.run("job-corrupt")
        finally:
            process_voice_clone.pop_request()

        db = _TestSession()
        job = db.get(Job, "job-corrupt")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error_code == "audio_corrupt"
        assert job.error_message == "The audio file is corrupt or in an unsupported format"
        db.close()

    def test_processing_error_sets_error_code(self, _mock_factory) -> None:
        """ProcessingError sets error_code='processing_error'."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-procerr")

        # retries=2 (at max) so retry is not attempted
        process_voice_clone.push_request(retries=2)
        try:
            with patch(
                "app.workers.tasks.build_speaker_profile",
                side_effect=ProcessingError("GPU failed", stage="separation"),
            ):
                process_voice_clone.run("job-procerr")
        finally:
            process_voice_clone.pop_request()

        db = _TestSession()
        job = db.get(Job, "job-procerr")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error_code == "processing_error"
        assert job.error_message == "An error occurred during processing"
        db.close()

    def test_generic_exception_sets_processing_error_code(self, _mock_factory) -> None:
        """Non-pipeline exceptions get error_code='processing_error' with raw message."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-generic")

        process_voice_clone.push_request(retries=2)
        try:
            with patch(
                "app.workers.tasks.build_speaker_profile",
                side_effect=RuntimeError("Unexpected crash"),
            ):
                process_voice_clone.run("job-generic")
        finally:
            process_voice_clone.pop_request()

        db = _TestSession()
        job = db.get(Job, "job-generic")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert job.error_code == "processing_error"
        assert job.error_message is not None
        assert "RuntimeError" in job.error_message
        assert "Unexpected crash" in job.error_message
        db.close()

    def test_no_speech_error_does_not_retry(self, _mock_factory) -> None:
        """NoSpeechDetectedError should NOT trigger retry (user-actionable)."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-no-retry-speech")

        mock_retry = MagicMock(side_effect=ValueError("should not be called"))

        process_voice_clone.push_request(retries=0)
        try:
            with (
                patch(
                    "app.workers.tasks.build_speaker_profile",
                    side_effect=NoSpeechDetectedError(),
                ),
                patch.object(
                    type(process_voice_clone._get_current_object()),
                    "retry",
                    mock_retry,
                ),
            ):
                process_voice_clone.run("job-no-retry-speech")
        finally:
            process_voice_clone.pop_request()

        mock_retry.assert_not_called()

    def test_insufficient_audio_does_not_retry(self, _mock_factory) -> None:
        """InsufficientAudioError should NOT trigger retry."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-no-retry-short")

        mock_retry = MagicMock(side_effect=ValueError("should not be called"))

        process_voice_clone.push_request(retries=0)
        try:
            with (
                patch(
                    "app.workers.tasks.build_speaker_profile",
                    side_effect=InsufficientAudioError(),
                ),
                patch.object(
                    type(process_voice_clone._get_current_object()),
                    "retry",
                    mock_retry,
                ),
            ):
                process_voice_clone.run("job-no-retry-short")
        finally:
            process_voice_clone.pop_request()

        mock_retry.assert_not_called()

    def test_audio_corrupt_does_not_retry(self, _mock_factory) -> None:
        """AudioCorruptError should NOT trigger retry."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-no-retry-corrupt")

        mock_retry = MagicMock(side_effect=ValueError("should not be called"))

        process_voice_clone.push_request(retries=0)
        try:
            with (
                patch(
                    "app.workers.tasks.build_speaker_profile",
                    side_effect=AudioCorruptError(),
                ),
                patch.object(
                    type(process_voice_clone._get_current_object()),
                    "retry",
                    mock_retry,
                ),
            ):
                process_voice_clone.run("job-no-retry-corrupt")
        finally:
            process_voice_clone.pop_request()

        mock_retry.assert_not_called()

    def test_processing_error_retries(self) -> None:
        """ProcessingError (retryable) triggers retry when retries remain."""
        from app.workers.tasks import process_voice_clone

        _create_job("job-retry-proc")

        mock_retry = MagicMock(side_effect=ValueError("retry triggered"))

        process_voice_clone.push_request(retries=0)
        try:
            with (
                patch("app.workers.tasks.get_session_factory", return_value=_TestSession),
                patch(
                    "app.workers.tasks.build_speaker_profile",
                    side_effect=ProcessingError("Transient failure", stage="separation"),
                ),
                patch.object(
                    type(process_voice_clone._get_current_object()),
                    "retry",
                    mock_retry,
                ),
                pytest.raises(ValueError, match="retry triggered"),
            ):
                process_voice_clone.run("job-retry-proc")
        finally:
            process_voice_clone.pop_request()

        mock_retry.assert_called_once()
