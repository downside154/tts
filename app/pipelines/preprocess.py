"""Audio preprocessing pipeline.

Includes VAD segmentation (Silero), background noise detection,
source separation (Demucs), speech enhancement (DeepFilterNet),
and loudness normalization (EBU R128).
"""

import logging
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.config import settings

logger = logging.getLogger(__name__)

SILERO_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION_MS = 250
MERGE_GAP_MS = 300
SNR_THRESHOLD_DB = 20.0


class Segment(NamedTuple):
    start: float
    end: float
    confidence: float


DEMUCS_SAMPLE_RATE = 44100
DEMUCS_MODEL_NAME = "htdemucs"

_vad_model = None
_vad_utils = None
_demucs_model = None


def _load_vad_model():
    """Lazily load and cache the Silero VAD model."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        logger.info("Loading Silero VAD model...")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        logger.info("Silero VAD model loaded")
    return _vad_model, _vad_utils


def _load_demucs_model():
    """Lazily load and cache the Demucs htdemucs model."""
    global _demucs_model
    if _demucs_model is None:
        from demucs.pretrained import get_model

        logger.info("Loading Demucs %s model...", DEMUCS_MODEL_NAME)
        _demucs_model = get_model(DEMUCS_MODEL_NAME)
        _demucs_model.eval()
        logger.info("Demucs model loaded")
    return _demucs_model


def separate_vocals(audio_path: Path) -> Path:
    """Separate vocals from background music/noise using Demucs.

    If the audio is clean (SNR >= 20dB), skips separation and returns
    the original path. Otherwise, runs htdemucs to isolate the vocals
    track and saves it as a WAV file.

    Args:
        audio_path: Path to the input audio file.

    Returns:
        Path to the isolated vocals WAV file, or the original path
        if separation was not needed.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not detect_needs_separation(audio_path):
        logger.info("Audio %s is clean — skipping separation", audio_path.name)
        return audio_path

    from demucs.apply import apply_model

    model = _load_demucs_model()
    device = torch.device(settings.device.value)

    # Load audio and resample to Demucs expected rate (44100 Hz)
    wav, orig_sr = torchaudio.load(str(audio_path))

    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # mono → stereo for htdemucs

    if orig_sr != DEMUCS_SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, orig_sr, DEMUCS_SAMPLE_RATE)

    # Demucs expects (batch, channels, samples)
    mix = wav.unsqueeze(0).to(device)
    model.to(device)

    with torch.no_grad():
        estimates = apply_model(model, mix, device=device)

    # htdemucs sources: ['drums', 'bass', 'other', 'vocals']
    vocals_idx = model.sources.index("vocals")
    vocals = estimates[0, vocals_idx]  # (channels, samples)

    # Convert back to mono
    vocals = vocals.mean(dim=0, keepdim=True).cpu()

    # Resample back to original sample rate if needed
    if orig_sr != DEMUCS_SAMPLE_RATE:
        vocals = torchaudio.functional.resample(vocals, DEMUCS_SAMPLE_RATE, orig_sr)

    # Save vocals to file
    output_path = audio_path.with_stem(audio_path.stem + "_vocals")
    torchaudio.save(str(output_path), vocals, orig_sr)

    logger.info(
        "Separated vocals from %s → %s",
        audio_path.name,
        output_path.name,
    )
    return output_path


def detect_speech_segments(audio_path: Path) -> list[Segment]:
    """Detect speech segments in an audio file using Silero VAD.

    Args:
        audio_path: Path to the input audio file (any format readable
            by torchaudio/soundfile).

    Returns:
        List of Segment namedtuples with start/end times in seconds
        and a confidence score (mean VAD probability for the segment).
        Returns an empty list if no speech is detected.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model, utils = _load_vad_model()
    get_speech_timestamps = utils[0]
    read_audio = utils[2]

    wav = read_audio(str(audio_path), sampling_rate=SILERO_SAMPLE_RATE)

    # Get speech timestamps in samples
    timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SILERO_SAMPLE_RATE,
        min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=MERGE_GAP_MS,
        return_seconds=False,
    )
    model.reset_states()

    if not timestamps:
        logger.warning("No speech detected in %s", audio_path)
        return []

    # Compute per-segment confidence from frame-level probabilities
    segments = []
    for ts in timestamps:
        start_sample = ts["start"]
        end_sample = ts["end"]
        start_s = start_sample / SILERO_SAMPLE_RATE
        end_s = end_sample / SILERO_SAMPLE_RATE

        confidence = _compute_segment_confidence(
            model, wav[start_sample:end_sample]
        )

        segments.append(Segment(start=start_s, end=end_s, confidence=confidence))

    logger.info(
        "Detected %d speech segment(s) in %s (total %.1fs)",
        len(segments),
        audio_path.name,
        sum(s.end - s.start for s in segments),
    )
    return segments


def detect_needs_separation(audio_path: Path) -> bool:
    """Determine if audio needs source separation based on SNR estimation.

    Uses VAD to identify speech vs. non-speech regions, then computes
    the signal-to-noise ratio. If estimated SNR < 20dB, the audio
    likely has significant background noise or music and needs separation.

    Args:
        audio_path: Path to the input audio file.

    Returns:
        True if the audio needs source separation (SNR < 20dB),
        False if the audio is clean enough to use directly.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    segments = detect_speech_segments(audio_path)

    if not segments:
        logger.info("No speech detected in %s — flagging for separation", audio_path.name)
        return True

    data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data[:, 0]

    total_samples = len(data)
    speech_mask = np.zeros(total_samples, dtype=bool)
    for seg in segments:
        start_idx = int(seg.start * sample_rate)
        end_idx = min(int(seg.end * sample_rate), total_samples)
        speech_mask[start_idx:end_idx] = True

    speech_samples = data[speech_mask]
    noise_samples = data[~speech_mask]

    if len(noise_samples) == 0:
        logger.info("No non-speech regions in %s — assuming clean", audio_path.name)
        return False

    signal_power = float(np.mean(speech_samples**2))
    noise_power = float(np.mean(noise_samples**2))

    if noise_power < 1e-10:
        logger.info("Noise floor near zero in %s — clean audio", audio_path.name)
        return False

    if signal_power < 1e-10:
        logger.info("Signal power near zero in %s — flagging for separation", audio_path.name)
        return True

    snr_db = 10.0 * math.log10(signal_power / noise_power)

    logger.info("Estimated SNR for %s: %.1f dB (threshold: %.1f dB)", audio_path.name, snr_db, SNR_THRESHOLD_DB)

    return snr_db < SNR_THRESHOLD_DB


def _compute_segment_confidence(
    model: torch.nn.Module,
    audio_chunk: torch.Tensor,
    window_size: int = 512,
) -> float:
    """Compute mean VAD probability for an audio chunk."""
    model.reset_states()  # type: ignore[operator]
    probs: list[float] = []

    for i in range(0, len(audio_chunk), window_size):
        chunk = audio_chunk[i : i + window_size]
        if len(chunk) < window_size:
            chunk = torch.nn.functional.pad(chunk, (0, window_size - len(chunk)))
        prob = model(chunk, SILERO_SAMPLE_RATE).item()
        probs.append(prob)

    model.reset_states()  # type: ignore[operator]
    return sum(probs) / len(probs) if probs else 0.0
