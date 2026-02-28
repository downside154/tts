"""Audio analysis pipeline.

Handles segment merging, quality scoring (SNR, clipping, duration),
and speaker profile construction from processed audio segments.
"""

import json
import logging
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
import torch

from app.config import settings
from app.errors import (
    AudioCorruptError,
    InsufficientAudioError,
    NoSpeechDetectedError,
    PipelineError,
    ProcessingError,
)
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


# ── Speaker embedding ─────────────────────────────────────────────

ECAPA_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SAMPLE_RATE = 16000

_ecapa_model = None


def _load_ecapa_model():
    """Lazily load and cache the ECAPA-TDNN speaker encoder."""
    global _ecapa_model
    if _ecapa_model is None:
        from app.pipelines.preprocess import _ensure_torchaudio_compat

        _ensure_torchaudio_compat()

        from speechbrain.inference.speaker import EncoderClassifier

        savedir = settings.storage_path / "models" / "spkrec-ecapa-voxceleb"
        logger.info("Loading ECAPA-TDNN model from %s...", ECAPA_MODEL_SOURCE)
        _ecapa_model = EncoderClassifier.from_hparams(
            source=ECAPA_MODEL_SOURCE,
            savedir=str(savedir),
        )
        logger.info("ECAPA-TDNN model loaded")
    return _ecapa_model


def compute_speaker_embedding(
    audio_path: Path,
    segments: list[ScoredSegment],
) -> np.ndarray:
    """Compute a 192-dim speaker embedding from selected audio segments.

    Concatenates the audio from the given segments and runs it through
    the ECAPA-TDNN encoder.  Audio is resampled to 16 kHz if needed.

    Args:
        audio_path: Path to the full audio file.
        segments: Selected scored segments to use for embedding.

    Returns:
        NumPy array of shape ``(192,)`` with the speaker embedding.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    chunks = []
    for seg in segments:
        start_idx = int(seg.start * sr)
        end_idx = min(int(seg.end * sr), len(data))
        chunks.append(data[start_idx:end_idx])

    concatenated = np.concatenate(chunks)
    signal = torch.tensor(concatenated).unsqueeze(0)  # (1, samples)

    if sr != ECAPA_SAMPLE_RATE:
        import torchaudio

        signal = torchaudio.functional.resample(signal, sr, ECAPA_SAMPLE_RATE)

    model = _load_ecapa_model()
    with torch.no_grad():
        embedding = model.encode_batch(signal)

    return embedding.squeeze().cpu().numpy()


# ── Speaker profile builder ───────────────────────────────────────


def _compute_snr_db(chunk: np.ndarray) -> float:
    """Compute SNR in decibels for an audio chunk.

    Uses the same frame-based approach as ``_snr_proxy_score`` but
    returns the raw dB value instead of a 0–1 score.
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

    n_quiet = max(1, len(energies) // 10)
    noise_energy = sum(energies[:n_quiet]) / n_quiet
    signal_energy = sum(energies) / len(energies)

    if signal_energy < 1e-10 or noise_energy < 1e-10:
        return 0.0

    return 10.0 * math.log10(signal_energy / noise_energy)


def _compute_quality_summary(
    data: np.ndarray,
    sr: int,
    segments: list[ScoredSegment],
) -> dict:
    """Compute quality summary statistics for selected segments."""
    snr_values: list[float] = []
    clipped_count = 0

    for seg in segments:
        start_idx = int(seg.start * sr)
        end_idx = min(int(seg.end * sr), len(data))
        chunk = data[start_idx:end_idx]

        snr_values.append(_compute_snr_db(chunk))

        if np.any(np.abs(chunk) > CLIPPING_THRESHOLD):
            clipped_count += 1

    mean_snr = sum(snr_values) / len(snr_values) if snr_values else 0.0
    return {
        "mean_snr": round(mean_snr, 1),
        "clipped_segments": clipped_count,
    }


MIN_SPEECH_DURATION = 3.0


def build_speaker_profile(audio_path: Path, job_id: str) -> dict:
    """Build a complete speaker profile from an audio file.

    Orchestrates the full analysis pipeline:
    noise detection, source separation, speech enhancement,
    diarization, segment scoring, embedding generation, and
    profile assembly.

    Args:
        audio_path: Path to the input audio file.
        job_id: Job identifier for organising output files.

    Returns:
        Dict matching the speaker-profile JSON schema, with keys:
        ``id``, ``created_at``, ``source_file``, ``embedding_path``,
        ``segments``, ``total_duration_s``, ``speaker_count``,
        ``dominant_speaker_id``, ``quality_summary``.

    Raises:
        AudioCorruptError: If the audio file cannot be decoded.
        NoSpeechDetectedError: If no speech is found in the audio.
        InsufficientAudioError: If total usable speech is < 3 seconds.
        ProcessingError: If an unexpected error occurs during processing.
    """
    from app.pipelines.diarize import diarize_speakers, select_dominant_speaker
    from app.pipelines.preprocess import enhance_speech, separate_vocals

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise AudioCorruptError(
            f"Audio file not found: {audio_path}",
            stage="input_validation",
        )

    # Validate the file is readable audio
    try:
        sf.info(str(audio_path))
    except Exception as exc:
        raise AudioCorruptError(
            f"Cannot read audio file '{audio_path.name}': {exc}",
            stage="input_validation",
        ) from exc

    # Create output directories
    output_dir = settings.storage_path / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    # 1. Source separation (if needed)
    try:
        processed_path = separate_vocals(audio_path)
    except PipelineError:
        raise
    except Exception as exc:
        raise ProcessingError(
            f"Source separation failed: {exc}",
            stage="separation",
        ) from exc

    # 2. Speech enhancement
    try:
        processed_path = enhance_speech(processed_path)
    except PipelineError:
        raise
    except Exception as exc:
        raise ProcessingError(
            f"Speech enhancement failed: {exc}",
            stage="enhancement",
        ) from exc

    # 3. Diarization
    try:
        speaker_segments = diarize_speakers(processed_path)
    except PipelineError:
        raise
    except Exception as exc:
        raise ProcessingError(
            f"Speaker diarization failed: {exc}",
            stage="diarization",
        ) from exc

    if not speaker_segments:
        raise NoSpeechDetectedError(stage="diarization")

    speaker_count = len({s.speaker_id for s in speaker_segments})

    # 4. Select dominant speaker and filter
    dominant_id = select_dominant_speaker(speaker_segments)
    dominant_segments = [
        s for s in speaker_segments if s.speaker_id == dominant_id
    ]

    # 5. Merge segments
    merged = merge_segments(dominant_segments)

    # 6. Score segments
    scored = [score_segment(processed_path, seg) for seg in merged]

    # 7. Select best segments
    selected = select_best_segments(scored)

    # 8. Check total duration is sufficient
    total_duration = sum(s.end - s.start for s in selected)
    if total_duration < MIN_SPEECH_DURATION:
        raise InsufficientAudioError(
            f"Only {total_duration:.1f}s of usable speech found "
            f"(minimum {MIN_SPEECH_DURATION:.0f}s required)",
            stage="segment_selection",
        )

    # 9. Extract segment WAVs
    data, sr = sf.read(str(processed_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    segment_infos: list[dict] = []
    for i, seg in enumerate(selected):
        seg_path = segments_dir / f"segment_{i:03d}.wav"
        start_idx = int(seg.start * sr)
        end_idx = min(int(seg.end * sr), len(data))
        sf.write(str(seg_path), data[start_idx:end_idx], sr, subtype="PCM_16")
        segment_infos.append({
            "path": str(seg_path),
            "start": seg.start,
            "end": seg.end,
            "quality_score": seg.quality_score,
        })

    # 10. Compute speaker embedding
    try:
        embedding = compute_speaker_embedding(processed_path, selected)
    except PipelineError:
        raise
    except Exception as exc:
        raise ProcessingError(
            f"Speaker embedding computation failed: {exc}",
            stage="embedding",
        ) from exc

    embedding_path = output_dir / "embedding.npy"
    np.save(str(embedding_path), embedding)

    # 11. Quality summary
    quality_summary = _compute_quality_summary(data, sr, selected)

    # 12. Assemble profile
    profile = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "source_file": audio_path.name,
        "embedding_path": str(embedding_path),
        "segments": segment_infos,
        "total_duration_s": round(total_duration, 1),
        "speaker_count": speaker_count,
        "dominant_speaker_id": dominant_id,
        "quality_summary": quality_summary,
    }

    # Save JSON
    profile_path = output_dir / "speaker_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    logger.info(
        "Built speaker profile %s: %d segment(s), %.1fs total",
        profile["id"],
        len(selected),
        total_duration,
    )
    return profile
