"""Tests for the audio ingestion pipeline.

Covers FFmpeg audio extraction from various input formats
(MP4, MP3, WAV) and error handling for corrupt files.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from app.pipelines.ingest import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    TARGET_SUBTYPE,
    AudioExtractionError,
    CorruptFileError,
    extract_audio,
)


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    """Create a temporary fixtures directory."""
    d = tmp_path / "fixtures"
    d.mkdir()
    return d


def _generate_sine_wav(
    path: Path,
    duration: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 1,
    freq: float = 440.0,
) -> Path:
    """Generate a sine wave WAV file for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 2:
        signal = np.column_stack([signal, signal])
    sf.write(str(path), signal, sample_rate, subtype="PCM_16")
    return path


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> Path:
    """Convert a WAV file to MP3 using FFmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-q:a", "2", str(mp3_path)],
        capture_output=True,
        check=True,
    )
    return mp3_path


def _wav_to_mp4(wav_path: Path, mp4_path: Path) -> Path:
    """Create an MP4 (video) with audio from a WAV file using FFmpeg.

    Generates a black video stream + the audio track.
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
            "-i", str(wav_path),
            "-shortest",
            "-c:v", "libx264", "-c:a", "aac",
            str(mp4_path),
        ],
        capture_output=True,
        check=True,
    )
    return mp4_path


def _assert_normalized_output(output_path: Path) -> None:
    """Assert the output file meets the normalization spec."""
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    info = sf.info(str(output_path))
    assert info.channels == TARGET_CHANNELS
    assert info.samplerate == TARGET_SAMPLE_RATE
    assert info.subtype == TARGET_SUBTYPE


class TestExtractAudioFromMP4:
    def test_mp4_to_mono_wav(self, fixtures_dir: Path) -> None:
        """MP4 video → mono WAV 24kHz 16-bit output."""
        wav_src = _generate_sine_wav(fixtures_dir / "src.wav")
        mp4_path = _wav_to_mp4(wav_src, fixtures_dir / "test_video.mp4")

        result = extract_audio(mp4_path, output_dir=fixtures_dir)

        _assert_normalized_output(result)


class TestExtractAudioFromMP3:
    def test_mp3_to_mono_wav(self, fixtures_dir: Path) -> None:
        """MP3 audio → mono WAV 24kHz 16-bit output."""
        wav_src = _generate_sine_wav(fixtures_dir / "src.wav")
        mp3_path = _wav_to_mp3(wav_src, fixtures_dir / "test_audio.mp3")

        result = extract_audio(mp3_path, output_dir=fixtures_dir)

        _assert_normalized_output(result)


class TestExtractAudioFromWAV:
    def test_stereo_wav_to_mono(self, fixtures_dir: Path) -> None:
        """WAV stereo → mono WAV 24kHz 16-bit output."""
        stereo_path = _generate_sine_wav(
            fixtures_dir / "stereo.wav",
            sample_rate=48000,
            channels=2,
        )

        result = extract_audio(stereo_path, output_dir=fixtures_dir)

        _assert_normalized_output(result)

    def test_already_normalized_wav(self, fixtures_dir: Path) -> None:
        """WAV already at 24kHz mono 16-bit → still produces valid output."""
        wav_path = _generate_sine_wav(
            fixtures_dir / "already_ok.wav",
            sample_rate=24000,
            channels=1,
        )

        result = extract_audio(wav_path, output_dir=fixtures_dir)

        _assert_normalized_output(result)

    def test_high_sample_rate_wav(self, fixtures_dir: Path) -> None:
        """WAV at 96kHz → downsampled to 24kHz."""
        wav_path = _generate_sine_wav(
            fixtures_dir / "high_sr.wav",
            sample_rate=96000,
            channels=1,
        )

        result = extract_audio(wav_path, output_dir=fixtures_dir)

        _assert_normalized_output(result)


class TestExtractAudioErrorHandling:
    def test_file_not_found(self, fixtures_dir: Path) -> None:
        """Non-existent file → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_audio(fixtures_dir / "nonexistent.wav")

    def test_corrupt_file(self, fixtures_dir: Path) -> None:
        """Corrupt file → CorruptFileError with descriptive message."""
        corrupt_path = fixtures_dir / "corrupt.wav"
        corrupt_path.write_bytes(b"this is not audio data at all")

        with pytest.raises(CorruptFileError, match="corrupt or has an unsupported format"):
            extract_audio(corrupt_path, output_dir=fixtures_dir)

    def test_empty_file(self, fixtures_dir: Path) -> None:
        """Empty file → raises an error."""
        empty_path = fixtures_dir / "empty.wav"
        empty_path.write_bytes(b"")

        with pytest.raises((CorruptFileError, AudioExtractionError)):
            extract_audio(empty_path, output_dir=fixtures_dir)

    def test_text_file_with_audio_extension(self, fixtures_dir: Path) -> None:
        """Text file renamed to .mp3 → raises error."""
        fake_path = fixtures_dir / "fake.mp3"
        fake_path.write_text("this is plain text, not audio")

        with pytest.raises((CorruptFileError, AudioExtractionError)):
            extract_audio(fake_path, output_dir=fixtures_dir)


class TestExtractAudioFFmpegErrors:
    def test_ffmpeg_not_found(self, fixtures_dir: Path) -> None:
        """Missing FFmpeg binary raises AudioExtractionError."""
        wav_path = _generate_sine_wav(fixtures_dir / "input.wav")

        with patch("app.pipelines.ingest.settings") as mock_settings:
            mock_settings.ffmpeg_path = "/nonexistent/ffmpeg"
            with pytest.raises(AudioExtractionError, match="FFmpeg not found"):
                extract_audio(wav_path, output_dir=fixtures_dir)

    def test_ffmpeg_timeout(self, fixtures_dir: Path) -> None:
        """FFmpeg timeout raises AudioExtractionError."""
        wav_path = _generate_sine_wav(fixtures_dir / "input.wav")

        with patch("app.pipelines.ingest.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)
            with pytest.raises(AudioExtractionError, match="timed out"):
                extract_audio(wav_path, output_dir=fixtures_dir)

    def test_ffmpeg_non_corrupt_failure(self, fixtures_dir: Path) -> None:
        """FFmpeg failure that is not a corrupt file error raises AudioExtractionError."""
        wav_path = _generate_sine_wav(fixtures_dir / "input.wav")

        with patch("app.pipelines.ingest.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["ffmpeg"],
                returncode=1,
                stdout="",
                stderr="Some unknown FFmpeg error occurred",
            )
            with pytest.raises(AudioExtractionError, match="FFmpeg failed"):
                extract_audio(wav_path, output_dir=fixtures_dir)


class TestValidateOutput:
    def test_missing_output_file(self, fixtures_dir: Path) -> None:
        """Validation fails when output file doesn't exist."""
        from app.pipelines.ingest import _validate_output

        with pytest.raises(AudioExtractionError, match="no output file"):
            _validate_output(fixtures_dir / "nonexistent.wav")

    def test_empty_output_file(self, fixtures_dir: Path) -> None:
        """Validation fails when output file is empty."""
        from app.pipelines.ingest import _validate_output

        empty_path = fixtures_dir / "empty.wav"
        empty_path.write_bytes(b"")

        with pytest.raises(AudioExtractionError, match="empty output file"):
            _validate_output(empty_path)

    def test_invalid_audio_output(self, fixtures_dir: Path) -> None:
        """Validation fails when output is not valid audio."""
        from app.pipelines.ingest import _validate_output

        bad_path = fixtures_dir / "bad.wav"
        bad_path.write_bytes(b"not audio data")

        with pytest.raises(AudioExtractionError, match="not a valid audio file"):
            _validate_output(bad_path)

    def test_wrong_channel_count(self, fixtures_dir: Path) -> None:
        """Validation fails when output has wrong number of channels."""
        from app.pipelines.ingest import _validate_output

        stereo_path = _generate_sine_wav(
            fixtures_dir / "stereo.wav",
            sample_rate=TARGET_SAMPLE_RATE,
            channels=2,
        )

        with pytest.raises(AudioExtractionError, match="channel"):
            _validate_output(stereo_path)

    def test_wrong_sample_rate(self, fixtures_dir: Path) -> None:
        """Validation fails when output has wrong sample rate."""
        from app.pipelines.ingest import _validate_output

        wrong_sr_path = _generate_sine_wav(
            fixtures_dir / "wrong_sr.wav",
            sample_rate=44100,
            channels=1,
        )

        with pytest.raises(AudioExtractionError, match="sample rate"):
            _validate_output(wrong_sr_path)

    def test_wrong_subtype(self, fixtures_dir: Path) -> None:
        """Validation fails when output has wrong subtype."""
        from app.pipelines.ingest import _validate_output

        path = fixtures_dir / "wrong_subtype.wav"
        t = np.linspace(0, 1.0, TARGET_SAMPLE_RATE, endpoint=False)
        signal = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(str(path), signal, TARGET_SAMPLE_RATE, subtype="FLOAT")

        with pytest.raises(AudioExtractionError, match="subtype"):
            _validate_output(path)


class TestExtractAudioOutputLocation:
    def test_default_output_dir(self, fixtures_dir: Path) -> None:
        """Output goes to same directory as input by default."""
        wav_path = _generate_sine_wav(fixtures_dir / "input.wav")

        result = extract_audio(wav_path)

        assert result.parent == fixtures_dir
        _assert_normalized_output(result)

    def test_custom_output_dir(self, fixtures_dir: Path) -> None:
        """Output goes to specified directory."""
        wav_path = _generate_sine_wav(fixtures_dir / "input.wav")
        out_dir = fixtures_dir / "output"

        result = extract_audio(wav_path, output_dir=out_dir)

        assert result.parent == out_dir
        _assert_normalized_output(result)
