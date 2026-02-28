"""Tests for the audio preprocessing pipeline.

Covers VAD segmentation, noise detection, source separation,
and speech enhancement functionality.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.pipelines.preprocess import (
    Segment,
    detect_needs_separation,
    detect_speech_segments,
)


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    d = tmp_path / "fixtures"
    d.mkdir()
    return d


def _generate_silence(path: Path, duration: float = 2.0, sample_rate: int = 16000) -> Path:
    """Generate a silent WAV file."""
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return path


def _generate_speech(path: Path, text: str = "Hello, this is a test") -> Path:
    """Generate speech audio using macOS say command."""
    subprocess.run(
        ["say", "-o", str(path), "--data-format=LEI16@16000", text],
        capture_output=True,
        check=True,
    )
    return path


def _generate_speech_with_silence(
    path: Path,
    silence_before: float = 1.0,
    silence_after: float = 1.0,
    sample_rate: int = 16000,
) -> Path:
    """Generate speech audio padded with silence before and after."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        speech_path = Path(tmp.name)

    _generate_speech(speech_path, "This is a speech segment for testing")

    speech_data, sr = sf.read(str(speech_path))
    speech_path.unlink()

    before = np.zeros(int(sample_rate * silence_before), dtype=speech_data.dtype)
    after = np.zeros(int(sample_rate * silence_after), dtype=speech_data.dtype)
    combined = np.concatenate([before, speech_data, after])

    sf.write(str(path), combined, sample_rate, subtype="PCM_16")
    return path


class TestDetectSpeechSegments:
    def test_speech_with_silence_returns_correct_boundaries(self, fixtures_dir: Path) -> None:
        """Speech padded with silence returns segment boundaries within tolerance."""
        audio_path = _generate_speech_with_silence(
            fixtures_dir / "speech_silence.wav",
            silence_before=1.0,
            silence_after=1.0,
        )

        segments = detect_speech_segments(audio_path)

        assert len(segments) >= 1
        # Speech starts after ~1s of silence (within 100ms tolerance)
        assert segments[0].start >= 0.9, f"Speech start {segments[0].start} too early"
        # Speech should end before the trailing silence
        total_duration = sf.info(str(audio_path)).duration
        assert segments[-1].end <= total_duration - 0.5, (
            f"Speech end {segments[-1].end} too close to file end {total_duration}"
        )
        # All timestamps are floats in seconds
        for seg in segments:
            assert isinstance(seg.start, float)
            assert isinstance(seg.end, float)
            assert isinstance(seg.confidence, float)
            assert seg.start < seg.end

    def test_silent_audio_returns_empty_list(self, fixtures_dir: Path) -> None:
        """Pure silence returns an empty segment list."""
        audio_path = _generate_silence(fixtures_dir / "silence.wav", duration=3.0)

        segments = detect_speech_segments(audio_path)

        assert segments == []

    def test_continuous_speech_returns_single_segment(self, fixtures_dir: Path) -> None:
        """Continuous speech without long pauses returns a single segment."""
        audio_path = _generate_speech(
            fixtures_dir / "continuous.wav",
            text="This is continuous speech without any long pauses in between words",
        )

        segments = detect_speech_segments(audio_path)

        assert len(segments) == 1
        duration = sf.info(str(audio_path)).duration
        # Single segment should span most of the audio
        segment_duration = segments[0].end - segments[0].start
        assert segment_duration >= duration * 0.5

    def test_segment_is_named_tuple(self, fixtures_dir: Path) -> None:
        """Segments are Segment namedtuples with start, end, confidence."""
        audio_path = _generate_speech(fixtures_dir / "named_tuple.wav")

        segments = detect_speech_segments(audio_path)

        assert len(segments) >= 1
        seg = segments[0]
        assert isinstance(seg, Segment)
        assert hasattr(seg, "start")
        assert hasattr(seg, "end")
        assert hasattr(seg, "confidence")

    def test_confidence_is_positive(self, fixtures_dir: Path) -> None:
        """Detected speech segments have positive confidence scores."""
        audio_path = _generate_speech(fixtures_dir / "confidence.wav")

        segments = detect_speech_segments(audio_path)

        assert len(segments) >= 1
        for seg in segments:
            assert seg.confidence > 0.0

    def test_file_not_found_raises_error(self, fixtures_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            detect_speech_segments(fixtures_dir / "nonexistent.wav")

    def test_multiple_speech_bursts_with_long_gap(self, fixtures_dir: Path) -> None:
        """Two speech bursts separated by a long silence gap produce two segments."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
            p1, p2 = Path(tmp1.name), Path(tmp2.name)

        _generate_speech(p1, "First segment")
        _generate_speech(p2, "Second segment")

        speech1, sr = sf.read(str(p1))
        speech2, _ = sf.read(str(p2))
        p1.unlink()
        p2.unlink()

        # 2 seconds of silence between speech bursts (well above 300ms merge threshold)
        silence = np.zeros(int(sr * 2.0), dtype=speech1.dtype)
        combined = np.concatenate([speech1, silence, speech2])

        audio_path = fixtures_dir / "two_bursts.wav"
        sf.write(str(audio_path), combined, sr, subtype="PCM_16")

        segments = detect_speech_segments(audio_path)

        assert len(segments) == 2
        # Second segment should start after the silence gap
        gap = segments[1].start - segments[0].end
        assert gap >= 1.0, f"Gap between segments ({gap}s) too small"


def _generate_speech_with_noise(
    path: Path,
    noise_level: float = 0.1,
    sample_rate: int = 16000,
) -> Path:
    """Generate speech audio mixed with background noise."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        speech_path = Path(tmp.name)

    _generate_speech(speech_path, "This is a speech test with background noise added")

    speech_data, sr = sf.read(str(speech_path))
    speech_path.unlink()

    # Add white noise at the specified level
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_level, len(speech_data)).astype(speech_data.dtype)

    # Pad with silence+noise before and after to create non-speech regions
    pad_samples = int(sample_rate * 1.0)
    before_noise = rng.normal(0, noise_level, pad_samples).astype(speech_data.dtype)
    after_noise = rng.normal(0, noise_level, pad_samples).astype(speech_data.dtype)

    mixed = np.concatenate([before_noise, speech_data + noise, after_noise])
    # Clip to prevent overflow
    mixed = np.clip(mixed, -1.0, 1.0)

    sf.write(str(path), mixed, sr, subtype="PCM_16")
    return path


def _generate_tonal_audio(path: Path, duration: float = 3.0, sample_rate: int = 16000) -> Path:
    """Generate tonal/music-like audio with no speech (chords)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Combine multiple frequencies to simulate music
    signal = (
        0.3 * np.sin(2 * np.pi * 261.63 * t)  # C4
        + 0.3 * np.sin(2 * np.pi * 329.63 * t)  # E4
        + 0.3 * np.sin(2 * np.pi * 392.00 * t)  # G4
    ).astype(np.float32)
    sf.write(str(path), signal, sample_rate, subtype="PCM_16")
    return path


class TestDetectNeedsSeparation:
    def test_clean_speech_returns_false(self, fixtures_dir: Path) -> None:
        """Clean speech recording with silent background returns False."""
        audio_path = _generate_speech_with_silence(
            fixtures_dir / "clean_speech.wav",
            silence_before=1.0,
            silence_after=1.0,
        )

        result = detect_needs_separation(audio_path)

        assert result is False

    def test_speech_with_loud_noise_returns_true(self, fixtures_dir: Path) -> None:
        """Speech mixed with loud background noise returns True."""
        audio_path = _generate_speech_with_noise(
            fixtures_dir / "noisy_speech.wav",
            noise_level=0.3,
        )

        result = detect_needs_separation(audio_path)

        assert result is True

    def test_speech_with_ambient_noise_returns_true(self, fixtures_dir: Path) -> None:
        """Speech with moderate ambient noise returns True."""
        audio_path = _generate_speech_with_noise(
            fixtures_dir / "ambient_speech.wav",
            noise_level=0.05,
        )

        result = detect_needs_separation(audio_path)

        assert result is True

    def test_pure_music_no_speech_returns_true(self, fixtures_dir: Path) -> None:
        """Pure tonal/music audio with no speech returns True."""
        audio_path = _generate_tonal_audio(fixtures_dir / "music.wav")

        result = detect_needs_separation(audio_path)

        assert result is True

    def test_file_not_found_raises_error(self, fixtures_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            detect_needs_separation(fixtures_dir / "nonexistent.wav")

    def test_returns_bool(self, fixtures_dir: Path) -> None:
        """Function returns a boolean value."""
        audio_path = _generate_speech_with_silence(
            fixtures_dir / "bool_check.wav",
            silence_before=1.0,
            silence_after=1.0,
        )

        result = detect_needs_separation(audio_path)

        assert isinstance(result, bool)

    def test_continuous_speech_no_noise_regions_returns_false(self, fixtures_dir: Path) -> None:
        """Continuous speech filling entire audio (no non-speech regions) returns False."""
        from unittest.mock import patch

        from app.pipelines.preprocess import Segment

        audio_path = _generate_speech(fixtures_dir / "continuous_full.wav", text="Continuous speech filling audio")

        info = sf.info(str(audio_path))
        full_segment = [Segment(start=0.0, end=info.duration, confidence=0.9)]

        with patch("app.pipelines.preprocess.detect_speech_segments", return_value=full_segment):
            result = detect_needs_separation(audio_path)

        assert result is False

    def test_stereo_audio_is_handled(self, fixtures_dir: Path) -> None:
        """Stereo audio input is correctly handled (downmixed to mono)."""
        from unittest.mock import patch

        from app.pipelines.preprocess import Segment

        # Generate stereo audio
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        mono = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        audio_path = fixtures_dir / "stereo.wav"
        sf.write(str(audio_path), stereo, sr, subtype="PCM_16")

        # Mock VAD to return a segment in the middle
        segment = [Segment(start=0.5, end=1.5, confidence=0.9)]
        with patch("app.pipelines.preprocess.detect_speech_segments", return_value=segment):
            result = detect_needs_separation(audio_path)

        assert isinstance(result, bool)

    def test_near_zero_signal_power_returns_true(self, fixtures_dir: Path) -> None:
        """Audio with near-zero signal power in speech regions but noise elsewhere returns True."""
        from unittest.mock import patch

        from app.pipelines.preprocess import Segment

        sr = 16000
        duration = 3.0
        n_samples = int(sr * duration)

        # Create audio: noise in first and last second, near-silence in middle
        rng = np.random.default_rng(42)
        samples = np.zeros(n_samples, dtype=np.float32)
        # Non-speech regions (first and last second) have noise
        noise_region = int(sr * 1.0)
        samples[:noise_region] = rng.normal(0, 0.1, noise_region).astype(np.float32)
        samples[-noise_region:] = rng.normal(0, 0.1, noise_region).astype(np.float32)
        # Speech region (middle) is near-silent

        audio_path = fixtures_dir / "near_zero_signal.wav"
        sf.write(str(audio_path), samples, sr, subtype="FLOAT")

        # Mock VAD to mark the near-silent middle as "speech"
        segment = [Segment(start=1.0, end=2.0, confidence=0.5)]
        with patch("app.pipelines.preprocess.detect_speech_segments", return_value=segment):
            result = detect_needs_separation(audio_path)

        assert result is True
