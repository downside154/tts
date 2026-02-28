"""Audio ingestion pipeline.

Handles extraction of audio from video/audio files using FFmpeg,
converting to a standardized format (mono WAV, 24kHz, 16-bit PCM).
"""

import logging
import subprocess
from pathlib import Path

import soundfile as sf

from app.config import settings

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"


class AudioExtractionError(Exception):
    """Raised when FFmpeg fails to extract or convert audio."""


class CorruptFileError(AudioExtractionError):
    """Raised when the input file is corrupt or unreadable."""


def extract_audio(input_path: Path, output_dir: Path | None = None) -> Path:
    """Extract and normalize audio from any video/audio file.

    Converts the input to mono WAV, 24kHz, 16-bit PCM using FFmpeg.

    Args:
        input_path: Path to the input video or audio file.
        output_dir: Directory for the output file. Defaults to the same
            directory as the input file.

    Returns:
        Path to the normalized WAV output file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        CorruptFileError: If the input file cannot be decoded.
        AudioExtractionError: If FFmpeg fails for any other reason.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_dir is None:
        output_dir = input_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_normalized.wav"

    cmd = [
        settings.ffmpeg_path,
        "-y",
        "-i", str(input_path),
        "-vn",                        # strip video
        "-acodec", "pcm_s16le",       # 16-bit PCM
        "-ar", str(TARGET_SAMPLE_RATE),  # 24kHz
        "-ac", str(TARGET_CHANNELS),  # mono
        str(output_path),
    ]

    logger.info("Extracting audio: %s -> %s", input_path, output_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        raise AudioExtractionError(
            f"FFmpeg not found at '{settings.ffmpeg_path}'. "
            "Ensure FFmpeg is installed and the path is correct."
        )
    except subprocess.TimeoutExpired:
        raise AudioExtractionError(
            f"FFmpeg timed out processing '{input_path.name}'"
        )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if _is_corrupt_file_error(stderr):
            raise CorruptFileError(
                f"Input file is corrupt or has an unsupported format: "
                f"{input_path.name}"
            )
        raise AudioExtractionError(
            f"FFmpeg failed (exit code {result.returncode}) "
            f"for '{input_path.name}': {stderr[-500:]}"
        )

    _validate_output(output_path)

    logger.info("Audio extraction complete: %s", output_path)
    return output_path


def _is_corrupt_file_error(stderr: str) -> bool:
    """Check if FFmpeg stderr indicates a corrupt or unreadable file."""
    indicators = [
        "Invalid data found when processing input",
        "could not find codec",
        "does not contain any stream",
        "No such file or directory",
        "Invalid argument",
        "End of file",
        "error while decoding",
    ]
    return any(indicator.lower() in stderr.lower() for indicator in indicators)


def _validate_output(output_path: Path) -> None:
    """Validate the output WAV file has the expected format."""
    if not output_path.exists():
        raise AudioExtractionError("FFmpeg produced no output file")

    if output_path.stat().st_size == 0:
        raise AudioExtractionError("FFmpeg produced an empty output file")

    try:
        info = sf.info(str(output_path))
    except Exception as exc:
        raise AudioExtractionError(
            f"Output file is not a valid audio file: {exc}"
        )

    if info.channels != TARGET_CHANNELS:
        raise AudioExtractionError(
            f"Expected {TARGET_CHANNELS} channel(s), got {info.channels}"
        )

    if info.samplerate != TARGET_SAMPLE_RATE:
        raise AudioExtractionError(
            f"Expected {TARGET_SAMPLE_RATE}Hz sample rate, "
            f"got {info.samplerate}Hz"
        )

    if info.subtype != TARGET_SUBTYPE:
        raise AudioExtractionError(
            f"Expected {TARGET_SUBTYPE} subtype, got {info.subtype}"
        )
