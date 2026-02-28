"""Audio analysis pipeline.

Handles segment merging, quality scoring (SNR, clipping, duration),
and speaker profile construction from processed audio segments.
"""

import logging
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

from app.pipelines.diarize import SpeakerSegment

logger = logging.getLogger(__name__)

MIN_SEGMENT_DURATION = 3.0
MAX_SEGMENT_DURATION = 15.0
MIN_ACCEPTABLE_DURATION = 3.0
MAX_ACCEPTABLE_DURATION = 30.0
CLIPPING_THRESHOLD = 0.99
TARGET_TOTAL_MIN = 10.0
TARGET_TOTAL_MAX = 30.0


class ScoredSegment(NamedTuple):
    start: float
    end: float
    speaker_id: str
    quality_score: float


def merge_segments(
    segments: list[SpeakerSegment],
    min_duration: float = MIN_SEGMENT_DURATION,
    max_duration: float = MAX_SEGMENT_DURATION,
) -> list[SpeakerSegment]:
    """Merge consecutive same-speaker segments into clips of target duration.

    Adjacent segments from the same speaker are merged as long as the
    combined duration stays within *max_duration*.  Segments are never
    merged across speaker boundaries.  Segments shorter than
    *min_duration* that cannot be merged are kept as-is.

    Args:
        segments: Speaker segments sorted by start time.
        min_duration: Minimum desired clip length in seconds.
        max_duration: Maximum allowed clip length in seconds.

    Returns:
        List of merged SpeakerSegment namedtuples.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.start)
    merged: list[SpeakerSegment] = []

    current = sorted_segs[0]
    for seg in sorted_segs[1:]:
        same_speaker = seg.speaker_id == current.speaker_id
        combined_duration = seg.end - current.start

        if same_speaker and combined_duration <= max_duration:
            # Extend current segment
            current = SpeakerSegment(
                start=current.start,
                end=seg.end,
                speaker_id=current.speaker_id,
            )
        else:
            merged.append(current)
            current = seg

    merged.append(current)

    logger.info(
        "Merged %d segment(s) into %d clip(s)",
        len(segments),
        len(merged),
    )
    return merged


def score_segment(
    audio_path: Path,
    segment: SpeakerSegment,
) -> ScoredSegment:
    """Compute a composite quality score for a single audio segment.

    The score is the mean of three sub-scores (each in 0–1):

    * **SNR proxy** — ratio of RMS energy to noise-floor estimate.
    * **Clipping** — 1.0 if no samples exceed the clipping threshold,
      decreasing with the fraction of clipped samples.
    * **Duration** — 1.0 for segments in the acceptable range (3–30 s),
      scaled down for segments outside that range.

    Args:
        audio_path: Path to the audio file.
        segment: The speaker segment to score.

    Returns:
        A ScoredSegment with the composite quality score.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    start_idx = int(segment.start * sample_rate)
    end_idx = min(int(segment.end * sample_rate), len(data))
    chunk = data[start_idx:end_idx]

    snr_score = _snr_proxy_score(chunk)
    clip_score = _clipping_score(chunk)
    dur_score = _duration_score(segment.end - segment.start)

    quality = (snr_score + clip_score + dur_score) / 3.0

    return ScoredSegment(
        start=segment.start,
        end=segment.end,
        speaker_id=segment.speaker_id,
        quality_score=round(quality, 4),
    )


def select_best_segments(
    scored: list[ScoredSegment],
    target_min: float = TARGET_TOTAL_MIN,
    target_max: float = TARGET_TOTAL_MAX,
) -> list[ScoredSegment]:
    """Select highest-quality segments summing to the target duration range.

    Segments are ranked by quality score (descending) and greedily
    selected until the total duration reaches *target_min*.  Selection
    stops once *target_max* is reached.  If all segments together are
    shorter than *target_min*, all are returned.

    Args:
        scored: List of scored segments.
        target_min: Minimum total duration in seconds.
        target_max: Maximum total duration in seconds.

    Returns:
        Selected segments sorted by start time.
    """
    if not scored:
        return []

    ranked = sorted(scored, key=lambda s: s.quality_score, reverse=True)

    selected: list[ScoredSegment] = []
    total = 0.0

    for seg in ranked:
        duration = seg.end - seg.start
        if total + duration > target_max and total >= target_min:
            break
        selected.append(seg)
        total += duration
        if total >= target_min:
            break

    # Return in chronological order
    selected.sort(key=lambda s: s.start)

    logger.info(
        "Selected %d segment(s) (%.1fs total) from %d candidates",
        len(selected),
        total,
        len(scored),
    )
    return selected


# ── Sub-score helpers ──────────────────────────────────────────────


def _snr_proxy_score(chunk: np.ndarray) -> float:
    """Estimate an SNR-like quality proxy from the audio chunk.

    Computes the ratio of RMS energy to the energy of the quietest
    10 % of frames (as a noise-floor estimate).  The resulting dB
    value is clamped and mapped to 0–1.
    """
    if len(chunk) == 0:
        return 0.0

    frame_size = 1024
    frames = [
        chunk[i : i + frame_size]
        for i in range(0, len(chunk), frame_size)
        if len(chunk[i : i + frame_size]) == frame_size
    ]

    if not frames:
        return 0.0

    energies = [float(np.mean(f**2)) for f in frames]
    energies.sort()

    # Noise-floor = mean energy of quietest 10 %
    n_quiet = max(1, len(energies) // 10)
    noise_energy = sum(energies[:n_quiet]) / n_quiet
    signal_energy = sum(energies) / len(energies)

    if signal_energy < 1e-10:
        return 0.0
    if noise_energy < 1e-10:
        return 1.0

    snr_db = 10.0 * math.log10(signal_energy / noise_energy)
    # Map 0–40 dB → 0–1
    return max(0.0, min(1.0, snr_db / 40.0))


def _clipping_score(chunk: np.ndarray) -> float:
    """Score based on the fraction of clipped samples.

    Returns 1.0 when no samples exceed the clipping threshold,
    decreasing linearly with the clipped fraction.
    """
    if len(chunk) == 0:
        return 0.0

    clipped = np.sum(np.abs(chunk) > CLIPPING_THRESHOLD)
    fraction = float(clipped) / len(chunk)
    return max(0.0, 1.0 - fraction * 10.0)


def _duration_score(duration: float) -> float:
    """Score based on segment duration.

    Full score for segments in the acceptable range (3–30 s).
    Linearly scaled down outside that range.
    """
    if duration <= 0:
        return 0.0
    if MIN_ACCEPTABLE_DURATION <= duration <= MAX_ACCEPTABLE_DURATION:
        return 1.0
    if duration < MIN_ACCEPTABLE_DURATION:
        return duration / MIN_ACCEPTABLE_DURATION
    # duration > MAX_ACCEPTABLE_DURATION
    return max(0.0, 1.0 - (duration - MAX_ACCEPTABLE_DURATION) / MAX_ACCEPTABLE_DURATION)
