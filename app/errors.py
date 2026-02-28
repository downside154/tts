"""Typed error classes for the analysis pipeline.

Each error carries an ``error_code`` (short machine-readable identifier)
and a ``user_message`` (human-readable explanation suitable for API
responses).
"""


class PipelineError(Exception):
    """Base class for all pipeline processing errors.

    Attributes:
        error_code: Short machine-readable identifier (e.g. ``"no_speech"``).
        user_message: Human-readable message suitable for API responses.
        stage: Pipeline stage where the error occurred (optional).
    """

    error_code: str = "processing_error"
    user_message: str = "An error occurred during processing"

    def __init__(
        self,
        message: str | None = None,
        *,
        stage: str | None = None,
    ) -> None:
        self.stage = stage
        super().__init__(message or self.user_message)


class NoSpeechDetectedError(PipelineError):
    """Raised when VAD or diarization finds no speech in the audio."""

    error_code: str = "no_speech"
    user_message: str = "No speech detected in the uploaded file"


class InsufficientAudioError(PipelineError):
    """Raised when total usable speech is less than 3 seconds."""

    error_code: str = "insufficient_audio"
    user_message: str = "At least 3 seconds of speech required"


class AudioCorruptError(PipelineError):
    """Raised when the audio file cannot be decoded or read."""

    error_code: str = "audio_corrupt"
    user_message: str = "The audio file is corrupt or in an unsupported format"


class ProcessingError(PipelineError):
    """Generic pipeline failure with stage information.

    Used as a catch-all for unexpected errors that occur during
    a specific pipeline stage.
    """

    error_code: str = "processing_error"
    user_message: str = "An error occurred during processing"
