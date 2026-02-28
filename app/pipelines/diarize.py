"""Speaker diarization pipeline.

Uses pyannote.audio for multi-speaker detection and segmentation,
identifying and selecting the dominant speaker from audio input.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch

from app.config import settings

logger = logging.getLogger(__name__)

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

_diarization_pipeline = None


class SpeakerSegment(NamedTuple):
    start: float
    end: float
    speaker_id: str


def _load_diarization_pipeline():
    """Lazily load and cache the pyannote diarization pipeline."""
    global _diarization_pipeline
    if _diarization_pipeline is None:
        from app.pipelines.preprocess import _ensure_torchaudio_compat

        _ensure_torchaudio_compat()

        from pyannote.audio import Pipeline

        logger.info("Loading diarization pipeline %s...", DIARIZATION_MODEL)
        _diarization_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=settings.hf_token or None,
        )
        device = torch.device(settings.device.value)
        _diarization_pipeline.to(device)
        logger.info("Diarization pipeline loaded")
    return _diarization_pipeline


def diarize_speakers(audio_path: Path) -> list[SpeakerSegment]:
    """Run speaker diarization on an audio file.

    Args:
        audio_path: Path to the input audio file.

    Returns:
        List of SpeakerSegment namedtuples with start/end times in
        seconds and a speaker_id string (e.g. ``"SPEAKER_00"``).
        Returns an empty list if no speakers are detected.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = _load_diarization_pipeline()
    diarization = pipeline(str(audio_path))

    segments: list[SpeakerSegment] = []
    for segment, _track, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            SpeakerSegment(
                start=segment.start,
                end=segment.end,
                speaker_id=speaker,
            )
        )

    speaker_ids = {s.speaker_id for s in segments}
    logger.info(
        "Diarized %s: %d segment(s), %d speaker(s)",
        audio_path.name,
        len(segments),
        len(speaker_ids),
    )
    return segments


def detect_multi_speaker(audio_path: Path) -> bool:
    """Detect whether audio contains multiple speakers.

    Args:
        audio_path: Path to the input audio file.

    Returns:
        True if multiple speakers are detected, False otherwise.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    segments = diarize_speakers(audio_path)
    speaker_ids = {s.speaker_id for s in segments}
    is_multi = len(speaker_ids) > 1
    logger.info(
        "Multi-speaker check for %s: %s (%d speaker(s))",
        Path(audio_path).name,
        is_multi,
        len(speaker_ids),
    )
    return is_multi


def select_dominant_speaker(segments: list[SpeakerSegment]) -> str:
    """Select the speaker with the most total speech time.

    Args:
        segments: List of SpeakerSegment namedtuples from diarization.

    Returns:
        The speaker_id of the dominant speaker.

    Raises:
        ValueError: If the segments list is empty.
    """
    if not segments:
        raise ValueError("Cannot select dominant speaker from empty segments")

    durations: dict[str, float] = defaultdict(float)
    for seg in segments:
        durations[seg.speaker_id] += seg.end - seg.start

    dominant = max(durations, key=lambda k: durations[k])
    logger.info(
        "Dominant speaker: %s (%.1fs of %.1fs total)",
        dominant,
        durations[dominant],
        sum(durations.values()),
    )
    return dominant
