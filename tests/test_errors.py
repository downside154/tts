"""Tests for typed pipeline error classes."""

import pytest

from app.errors import (
    AudioCorruptError,
    InsufficientAudioError,
    NoSpeechDetectedError,
    PipelineError,
    ProcessingError,
)


class TestPipelineError:
    def test_is_exception(self) -> None:
        assert issubclass(PipelineError, Exception)

    def test_default_message(self) -> None:
        err = PipelineError()
        assert str(err) == "An error occurred during processing"

    def test_custom_message(self) -> None:
        err = PipelineError("Custom failure")
        assert str(err) == "Custom failure"

    def test_error_code(self) -> None:
        err = PipelineError()
        assert err.error_code == "processing_error"

    def test_user_message(self) -> None:
        err = PipelineError()
        assert err.user_message == "An error occurred during processing"

    def test_stage_defaults_to_none(self) -> None:
        err = PipelineError()
        assert err.stage is None

    def test_stage_can_be_set(self) -> None:
        err = PipelineError(stage="diarization")
        assert err.stage == "diarization"


class TestNoSpeechDetectedError:
    def test_is_pipeline_error(self) -> None:
        assert issubclass(NoSpeechDetectedError, PipelineError)

    def test_error_code(self) -> None:
        err = NoSpeechDetectedError()
        assert err.error_code == "no_speech"

    def test_user_message(self) -> None:
        err = NoSpeechDetectedError()
        assert err.user_message == "No speech detected in the uploaded file"

    def test_default_str(self) -> None:
        err = NoSpeechDetectedError()
        assert str(err) == "No speech detected in the uploaded file"

    def test_custom_message(self) -> None:
        err = NoSpeechDetectedError("VAD returned zero segments")
        assert str(err) == "VAD returned zero segments"

    def test_stage(self) -> None:
        err = NoSpeechDetectedError(stage="diarization")
        assert err.stage == "diarization"

    def test_catchable_as_pipeline_error(self) -> None:
        with pytest.raises(PipelineError):
            raise NoSpeechDetectedError()


class TestInsufficientAudioError:
    def test_is_pipeline_error(self) -> None:
        assert issubclass(InsufficientAudioError, PipelineError)

    def test_error_code(self) -> None:
        err = InsufficientAudioError()
        assert err.error_code == "insufficient_audio"

    def test_user_message(self) -> None:
        err = InsufficientAudioError()
        assert err.user_message == "At least 3 seconds of speech required"

    def test_default_str(self) -> None:
        err = InsufficientAudioError()
        assert str(err) == "At least 3 seconds of speech required"

    def test_stage(self) -> None:
        err = InsufficientAudioError(stage="segment_selection")
        assert err.stage == "segment_selection"


class TestAudioCorruptError:
    def test_is_pipeline_error(self) -> None:
        assert issubclass(AudioCorruptError, PipelineError)

    def test_error_code(self) -> None:
        err = AudioCorruptError()
        assert err.error_code == "audio_corrupt"

    def test_user_message(self) -> None:
        err = AudioCorruptError()
        assert err.user_message == "The audio file is corrupt or in an unsupported format"

    def test_custom_message(self) -> None:
        err = AudioCorruptError("Cannot decode: invalid header")
        assert str(err) == "Cannot decode: invalid header"

    def test_stage(self) -> None:
        err = AudioCorruptError(stage="input_validation")
        assert err.stage == "input_validation"


class TestProcessingError:
    def test_is_pipeline_error(self) -> None:
        assert issubclass(ProcessingError, PipelineError)

    def test_error_code(self) -> None:
        err = ProcessingError()
        assert err.error_code == "processing_error"

    def test_user_message(self) -> None:
        err = ProcessingError()
        assert err.user_message == "An error occurred during processing"

    def test_custom_message(self) -> None:
        err = ProcessingError("CUDA out of memory")
        assert str(err) == "CUDA out of memory"

    def test_stage(self) -> None:
        err = ProcessingError(stage="embedding")
        assert err.stage == "embedding"
