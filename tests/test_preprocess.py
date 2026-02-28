"""Tests for the audio preprocessing pipeline.

Covers VAD segmentation, noise detection, source separation,
and speech enhancement functionality.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from app.pipelines.preprocess import (
    DEEPFILTER_SAMPLE_RATE,
    DEMUCS_SAMPLE_RATE,
    Segment,
    _ensure_torchaudio_compat,
    detect_needs_separation,
    detect_speech_segments,
    enhance_speech,
    separate_vocals,
)

# Install torchaudio.backend shim so df.enhance can be imported in tests
_ensure_torchaudio_compat()


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


def _make_mock_demucs_model(sources=None):
    """Create a mock Demucs model with proper sources attribute."""
    from unittest.mock import MagicMock

    model = MagicMock()
    model.sources = sources or ["drums", "bass", "other", "vocals"]
    model.to.return_value = model
    model.eval.return_value = model
    return model


class TestSeparateVocals:
    def test_clean_audio_skips_separation(self, fixtures_dir: Path) -> None:
        """Clean audio returns original path without running Demucs."""
        from unittest.mock import patch

        audio_path = fixtures_dir / "clean.wav"
        samples = np.zeros(16000, dtype=np.float32)
        sf.write(str(audio_path), samples, 16000, subtype="PCM_16")

        with patch("app.pipelines.preprocess.detect_needs_separation", return_value=False):
            result = separate_vocals(audio_path)

        assert result == audio_path

    def test_file_not_found_raises_error(self, fixtures_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            separate_vocals(fixtures_dir / "nonexistent.wav")

    def test_noisy_audio_runs_separation(self, fixtures_dir: Path) -> None:
        """Noisy audio triggers Demucs and produces a vocals file."""
        from unittest.mock import MagicMock, patch

        sr = 16000
        n_samples = sr * 2
        audio_path = fixtures_dir / "noisy.wav"
        samples = np.random.default_rng(42).normal(0, 0.1, n_samples).astype(np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        mock_model = _make_mock_demucs_model()

        # apply_model returns (batch=1, sources=4, channels=2, resampled_samples)
        resampled_len = int(n_samples * DEMUCS_SAMPLE_RATE / sr)
        fake_estimates = torch.randn(1, 4, 2, resampled_len)
        mock_apply = MagicMock(return_value=fake_estimates)

        with (
            patch("app.pipelines.preprocess.detect_needs_separation", return_value=True),
            patch("app.pipelines.preprocess._load_demucs_model", return_value=mock_model),
            patch("demucs.apply.apply_model", mock_apply),
        ):
            result = separate_vocals(audio_path)

        assert result != audio_path
        assert result.stem.endswith("_vocals")
        assert result.exists()

    def test_output_is_valid_wav(self, fixtures_dir: Path) -> None:
        """Output file is a valid WAV with correct sample rate."""
        from unittest.mock import MagicMock, patch

        sr = 16000
        n_samples = sr * 2
        audio_path = fixtures_dir / "valid_wav.wav"
        samples = np.random.default_rng(42).normal(0, 0.1, n_samples).astype(np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        mock_model = _make_mock_demucs_model()
        resampled_len = int(n_samples * DEMUCS_SAMPLE_RATE / sr)
        fake_estimates = torch.randn(1, 4, 2, resampled_len)
        mock_apply = MagicMock(return_value=fake_estimates)

        with (
            patch("app.pipelines.preprocess.detect_needs_separation", return_value=True),
            patch("app.pipelines.preprocess._load_demucs_model", return_value=mock_model),
            patch("demucs.apply.apply_model", mock_apply),
        ):
            result = separate_vocals(audio_path)

        info = sf.info(str(result))
        assert info.samplerate == sr
        assert info.channels == 1

    def test_mono_input_is_duplicated_to_stereo(self, fixtures_dir: Path) -> None:
        """Mono input is converted to stereo for htdemucs."""
        from unittest.mock import MagicMock, patch

        sr = 44100
        n_samples = sr * 1
        audio_path = fixtures_dir / "mono.wav"
        samples = np.random.default_rng(42).normal(0, 0.1, n_samples).astype(np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        mock_model = _make_mock_demucs_model()
        fake_estimates = torch.randn(1, 4, 2, n_samples)
        mock_apply = MagicMock(return_value=fake_estimates)

        with (
            patch("app.pipelines.preprocess.detect_needs_separation", return_value=True),
            patch("app.pipelines.preprocess._load_demucs_model", return_value=mock_model),
            patch("demucs.apply.apply_model", mock_apply),
        ):
            separate_vocals(audio_path)

        # Check that apply_model received stereo input (batch=1, channels=2, samples)
        call_args = mock_apply.call_args
        mix_tensor = call_args[0][1]
        assert mix_tensor.shape[1] == 2

    def test_stereo_input_not_duplicated(self, fixtures_dir: Path) -> None:
        """Stereo input passes through without channel duplication."""
        from unittest.mock import MagicMock, patch

        sr = 44100
        n_samples = sr * 1
        audio_path = fixtures_dir / "stereo.wav"
        mono = np.random.default_rng(42).normal(0, 0.1, n_samples).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        sf.write(str(audio_path), stereo, sr, subtype="PCM_16")

        mock_model = _make_mock_demucs_model()
        fake_estimates = torch.randn(1, 4, 2, n_samples)
        mock_apply = MagicMock(return_value=fake_estimates)

        with (
            patch("app.pipelines.preprocess.detect_needs_separation", return_value=True),
            patch("app.pipelines.preprocess._load_demucs_model", return_value=mock_model),
            patch("demucs.apply.apply_model", mock_apply),
        ):
            separate_vocals(audio_path)

        call_args = mock_apply.call_args
        mix_tensor = call_args[0][1]
        assert mix_tensor.shape[1] == 2

    def test_resampling_when_rate_differs(self, fixtures_dir: Path) -> None:
        """Audio at non-44100 rate is resampled for Demucs and resampled back."""
        from unittest.mock import MagicMock, patch

        sr = 16000
        n_samples = sr * 1
        audio_path = fixtures_dir / "resample.wav"
        samples = np.random.default_rng(42).normal(0, 0.1, n_samples).astype(np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        mock_model = _make_mock_demucs_model()
        resampled_len = int(n_samples * DEMUCS_SAMPLE_RATE / sr)
        fake_estimates = torch.randn(1, 4, 2, resampled_len)
        mock_apply = MagicMock(return_value=fake_estimates)

        with (
            patch("app.pipelines.preprocess.detect_needs_separation", return_value=True),
            patch("app.pipelines.preprocess._load_demucs_model", return_value=mock_model),
            patch("demucs.apply.apply_model", mock_apply),
        ):
            result = separate_vocals(audio_path)

        # Output should be at original sample rate
        info = sf.info(str(result))
        assert info.samplerate == sr

    def test_returns_path_object(self, fixtures_dir: Path) -> None:
        """Return value is a Path object."""
        from unittest.mock import MagicMock, patch

        sr = 16000
        audio_path = fixtures_dir / "path_check.wav"
        samples = np.zeros(sr, dtype=np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        mock_model = _make_mock_demucs_model()
        resampled_len = int(sr * DEMUCS_SAMPLE_RATE / sr)
        fake_estimates = torch.randn(1, 4, 2, resampled_len)
        mock_apply = MagicMock(return_value=fake_estimates)

        with (
            patch("app.pipelines.preprocess.detect_needs_separation", return_value=True),
            patch("app.pipelines.preprocess._load_demucs_model", return_value=mock_model),
            patch("demucs.apply.apply_model", mock_apply),
        ):
            result = separate_vocals(audio_path)

        assert isinstance(result, Path)

    def test_load_demucs_model_caches(self) -> None:
        """_load_demucs_model caches the model after first load."""
        from unittest.mock import MagicMock, patch

        import app.pipelines.preprocess as preprocess_mod

        original = preprocess_mod._demucs_model
        try:
            preprocess_mod._demucs_model = None

            mock_get_model = MagicMock()
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            with patch("demucs.pretrained.get_model", mock_get_model):
                result1 = preprocess_mod._load_demucs_model()
                result2 = preprocess_mod._load_demucs_model()

            assert result1 is result2
            mock_get_model.assert_called_once()
        finally:
            preprocess_mod._demucs_model = original


class TestEnhanceSpeech:
    def test_file_not_found_raises_error(self, fixtures_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            enhance_speech(fixtures_dir / "nonexistent.wav")

    def test_produces_enhanced_output(self, fixtures_dir: Path) -> None:
        """Enhancement produces an output file with _enhanced suffix."""
        from unittest.mock import MagicMock, patch

        sr = 48000
        n_samples = sr * 1
        audio_path = fixtures_dir / "speech.wav"
        samples = np.random.default_rng(42).normal(0, 0.1, n_samples).astype(np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        mock_model = MagicMock()
        mock_state = MagicMock()
        mock_enhance = MagicMock(side_effect=lambda m, s, audio: audio)

        with (
            patch("app.pipelines.preprocess._load_deepfilter_model", return_value=(mock_model, mock_state)),
            patch("df.enhance.enhance", mock_enhance),
        ):
            result = enhance_speech(audio_path)

        assert result.stem.endswith("_enhanced")
        assert result.exists()

    def test_output_is_valid_wav(self, fixtures_dir: Path) -> None:
        """Output is a valid WAV with correct sample rate and channels."""
        from unittest.mock import MagicMock, patch

        sr = 16000
        n_samples = sr * 1
        audio_path = fixtures_dir / "valid.wav"
        samples = np.random.default_rng(42).normal(0, 0.1, n_samples).astype(np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        # Mock enhance to return a tensor at DEEPFILTER_SAMPLE_RATE
        resampled_len = int(n_samples * DEEPFILTER_SAMPLE_RATE / sr)

        def fake_enhance(model, state, audio):
            return torch.randn(1, resampled_len)

        with (
            patch("app.pipelines.preprocess._load_deepfilter_model", return_value=(MagicMock(), MagicMock())),
            patch("df.enhance.enhance", side_effect=fake_enhance),
        ):
            result = enhance_speech(audio_path)

        info = sf.info(str(result))
        assert info.samplerate == sr
        assert info.channels == 1

    def test_no_resample_at_48khz(self, fixtures_dir: Path) -> None:
        """Audio already at 48kHz skips resampling."""
        from unittest.mock import MagicMock, patch

        sr = 48000
        n_samples = sr * 1
        audio_path = fixtures_dir / "at48k.wav"
        samples = np.random.default_rng(42).normal(0, 0.1, n_samples).astype(np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        mock_enhance = MagicMock(side_effect=lambda m, s, audio: audio)
        mock_resample = MagicMock()

        with (
            patch("app.pipelines.preprocess._load_deepfilter_model", return_value=(MagicMock(), MagicMock())),
            patch("df.enhance.enhance", mock_enhance),
            patch("app.pipelines.preprocess.torchaudio.functional.resample", mock_resample),
        ):
            enhance_speech(audio_path)

        # resample should not be called since audio is already at 48kHz
        mock_resample.assert_not_called()

    def test_resampling_when_rate_differs(self, fixtures_dir: Path) -> None:
        """Audio at non-48kHz is resampled for processing and back."""
        from unittest.mock import MagicMock, patch

        sr = 16000
        n_samples = sr * 1
        audio_path = fixtures_dir / "resample_df.wav"
        samples = np.random.default_rng(42).normal(0, 0.1, n_samples).astype(np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        resampled_len = int(n_samples * DEEPFILTER_SAMPLE_RATE / sr)
        mock_enhance = MagicMock(side_effect=lambda m, s, audio: torch.randn(1, resampled_len))

        with (
            patch("app.pipelines.preprocess._load_deepfilter_model", return_value=(MagicMock(), MagicMock())),
            patch("df.enhance.enhance", mock_enhance),
        ):
            result = enhance_speech(audio_path)

        info = sf.info(str(result))
        assert info.samplerate == sr

    def test_returns_path_object(self, fixtures_dir: Path) -> None:
        """Return value is a Path object."""
        from unittest.mock import MagicMock, patch

        sr = 48000
        audio_path = fixtures_dir / "path_type.wav"
        samples = np.zeros(sr, dtype=np.float32)
        sf.write(str(audio_path), samples, sr, subtype="PCM_16")

        mock_enhance = MagicMock(side_effect=lambda m, s, audio: audio)

        with (
            patch("app.pipelines.preprocess._load_deepfilter_model", return_value=(MagicMock(), MagicMock())),
            patch("df.enhance.enhance", mock_enhance),
        ):
            result = enhance_speech(audio_path)

        assert isinstance(result, Path)

    def test_load_deepfilter_model_caches(self) -> None:
        """_load_deepfilter_model caches model after first load."""
        from unittest.mock import MagicMock, patch

        import app.pipelines.preprocess as preprocess_mod

        orig_model = preprocess_mod._deepfilter_model
        orig_state = preprocess_mod._deepfilter_state
        try:
            preprocess_mod._deepfilter_model = None
            preprocess_mod._deepfilter_state = None

            mock_init = MagicMock(return_value=(MagicMock(), MagicMock(), "suffix"))

            with (
                patch("app.pipelines.preprocess._ensure_torchaudio_compat"),
                patch("df.enhance.init_df", mock_init),
            ):
                result1 = preprocess_mod._load_deepfilter_model()
                result2 = preprocess_mod._load_deepfilter_model()

            assert result1 == result2
            mock_init.assert_called_once()
        finally:
            preprocess_mod._deepfilter_model = orig_model
            preprocess_mod._deepfilter_state = orig_state

    def test_ensure_torchaudio_compat_shim(self, fixtures_dir: Path) -> None:
        """_ensure_torchaudio_compat installs all torchaudio shims."""
        import sys

        import torchaudio as ta

        from app.pipelines.preprocess import _ensure_torchaudio_compat

        # Save and remove existing shim attributes
        saved_attrs = {}
        for attr in ("AudioMetaData", "info", "list_audio_backends"):
            if hasattr(ta, attr):
                saved_attrs[attr] = getattr(ta, attr)
                delattr(ta, attr)

        saved_mods = {}
        for key in ["torchaudio.backend", "torchaudio.backend.common"]:
            if key in sys.modules:
                saved_mods[key] = sys.modules.pop(key)

        try:
            _ensure_torchaudio_compat()
            assert hasattr(ta, "AudioMetaData")
            assert hasattr(ta, "info")
            assert hasattr(ta, "list_audio_backends")
            assert "torchaudio.backend" in sys.modules

            # Verify info shim works with a real file
            audio_path = fixtures_dir / "compat_test.wav"
            samples = np.zeros(16000, dtype=np.float32)
            sf.write(str(audio_path), samples, 16000, subtype="PCM_16")
            meta = ta.info(str(audio_path))
            assert meta.sample_rate == 16000
            assert meta.num_channels == 1
            assert meta.bits_per_sample == 16

            # Verify list_audio_backends shim
            assert ta.list_audio_backends() == ["soundfile"]
        finally:
            for attr in ("AudioMetaData", "info", "list_audio_backends"):
                if hasattr(ta, attr):
                    delattr(ta, attr)
            for attr, val in saved_attrs.items():
                setattr(ta, attr, val)
            for key in ["torchaudio.backend", "torchaudio.backend.common"]:
                sys.modules.pop(key, None)
            sys.modules.update(saved_mods)

    def test_ensure_torchaudio_compat_idempotent(self) -> None:
        """Calling _ensure_torchaudio_compat twice doesn't overwrite shim."""
        import sys

        import torchaudio as ta

        from app.pipelines.preprocess import _ensure_torchaudio_compat

        saved_mods = {}
        for key in ["torchaudio.backend", "torchaudio.backend.common"]:
            if key in sys.modules:
                saved_mods[key] = sys.modules.pop(key)

        saved_meta = getattr(ta, "AudioMetaData", None)

        try:
            # Remove so the shim installs fresh
            if hasattr(ta, "AudioMetaData"):
                delattr(ta, "AudioMetaData")

            _ensure_torchaudio_compat()
            first_cls = ta.AudioMetaData
            first_module = sys.modules["torchaudio.backend"]

            _ensure_torchaudio_compat()
            assert ta.AudioMetaData is first_cls
            assert sys.modules["torchaudio.backend"] is first_module
        finally:
            for key in ["torchaudio.backend", "torchaudio.backend.common"]:
                sys.modules.pop(key, None)
            sys.modules.update(saved_mods)
            if saved_meta is not None:
                ta.AudioMetaData = saved_meta
