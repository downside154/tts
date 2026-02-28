"""Tests for the speaker diarization pipeline.

Covers diarize_speakers, detect_multi_speaker, and select_dominant_speaker.
All tests mock the pyannote pipeline to avoid heavy model loading.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from app.pipelines.preprocess import _ensure_torchaudio_compat

# Install torchaudio shim so pyannote.audio can be imported in tests
_ensure_torchaudio_compat()

from app.pipelines.diarize import (  # noqa: E402
    SpeakerSegment,
    detect_multi_speaker,
    diarize_speakers,
    select_dominant_speaker,
)


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    d = tmp_path / "fixtures"
    d.mkdir()
    return d


def _make_audio(path: Path, duration: float = 2.0, sr: int = 16000) -> Path:
    """Create a simple WAV file for testing."""
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(str(path), samples, sr, subtype="PCM_16")
    return path


def _make_mock_annotation(segments: list[tuple[float, float, str]]):
    """Create a mock pyannote Annotation from (start, end, speaker) tuples."""
    mock = MagicMock()

    class FakeSegment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    tracks = [(FakeSegment(s, e), "track", spk) for s, e, spk in segments]
    mock.itertracks.return_value = tracks
    mock.labels.return_value = sorted({spk for _, _, spk in segments})
    return mock


class TestDiarizeSpeakers:
    def test_single_speaker(self, fixtures_dir: Path) -> None:
        """Single-speaker audio returns all segments with the same speaker_id."""
        audio_path = _make_audio(fixtures_dir / "single.wav")
        annotation = _make_mock_annotation([
            (0.0, 1.5, "SPEAKER_00"),
            (2.0, 3.5, "SPEAKER_00"),
        ])
        mock_pipeline = MagicMock(return_value=annotation)

        with patch("app.pipelines.diarize._load_diarization_pipeline", return_value=mock_pipeline):
            segments = diarize_speakers(audio_path)

        assert len(segments) == 2
        assert all(s.speaker_id == "SPEAKER_00" for s in segments)

    def test_two_speakers(self, fixtures_dir: Path) -> None:
        """Two-speaker audio returns segments with distinct speaker_ids."""
        audio_path = _make_audio(fixtures_dir / "two_speakers.wav")
        annotation = _make_mock_annotation([
            (0.0, 2.0, "SPEAKER_00"),
            (2.5, 4.0, "SPEAKER_01"),
            (4.5, 6.0, "SPEAKER_00"),
        ])
        mock_pipeline = MagicMock(return_value=annotation)

        with patch("app.pipelines.diarize._load_diarization_pipeline", return_value=mock_pipeline):
            segments = diarize_speakers(audio_path)

        assert len(segments) == 3
        speaker_ids = {s.speaker_id for s in segments}
        assert speaker_ids == {"SPEAKER_00", "SPEAKER_01"}

    def test_returns_speaker_segments(self, fixtures_dir: Path) -> None:
        """Returned items are SpeakerSegment namedtuples."""
        audio_path = _make_audio(fixtures_dir / "named_tuple.wav")
        annotation = _make_mock_annotation([(0.0, 1.0, "SPEAKER_00")])
        mock_pipeline = MagicMock(return_value=annotation)

        with patch("app.pipelines.diarize._load_diarization_pipeline", return_value=mock_pipeline):
            segments = diarize_speakers(audio_path)

        assert len(segments) == 1
        seg = segments[0]
        assert isinstance(seg, SpeakerSegment)
        assert seg.start == 0.0
        assert seg.end == 1.0
        assert seg.speaker_id == "SPEAKER_00"

    def test_empty_diarization(self, fixtures_dir: Path) -> None:
        """Audio with no detected speakers returns empty list."""
        audio_path = _make_audio(fixtures_dir / "empty.wav")
        annotation = _make_mock_annotation([])
        mock_pipeline = MagicMock(return_value=annotation)

        with patch("app.pipelines.diarize._load_diarization_pipeline", return_value=mock_pipeline):
            segments = diarize_speakers(audio_path)

        assert segments == []

    def test_file_not_found_raises_error(self, fixtures_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            diarize_speakers(fixtures_dir / "nonexistent.wav")

    def test_segment_timestamps_are_floats(self, fixtures_dir: Path) -> None:
        """Segment start and end times are floats."""
        audio_path = _make_audio(fixtures_dir / "floats.wav")
        annotation = _make_mock_annotation([(0.5, 1.5, "SPEAKER_00")])
        mock_pipeline = MagicMock(return_value=annotation)

        with patch("app.pipelines.diarize._load_diarization_pipeline", return_value=mock_pipeline):
            segments = diarize_speakers(audio_path)

        for seg in segments:
            assert isinstance(seg.start, float)
            assert isinstance(seg.end, float)


class TestDetectMultiSpeaker:
    def test_single_speaker_returns_false(self, fixtures_dir: Path) -> None:
        """Single-speaker audio returns False."""
        audio_path = _make_audio(fixtures_dir / "single_ms.wav")
        annotation = _make_mock_annotation([
            (0.0, 1.0, "SPEAKER_00"),
            (2.0, 3.0, "SPEAKER_00"),
        ])
        mock_pipeline = MagicMock(return_value=annotation)

        with patch("app.pipelines.diarize._load_diarization_pipeline", return_value=mock_pipeline):
            result = detect_multi_speaker(audio_path)

        assert result is False

    def test_two_speakers_returns_true(self, fixtures_dir: Path) -> None:
        """Two-speaker audio returns True."""
        audio_path = _make_audio(fixtures_dir / "two_ms.wav")
        annotation = _make_mock_annotation([
            (0.0, 1.0, "SPEAKER_00"),
            (1.5, 3.0, "SPEAKER_01"),
        ])
        mock_pipeline = MagicMock(return_value=annotation)

        with patch("app.pipelines.diarize._load_diarization_pipeline", return_value=mock_pipeline):
            result = detect_multi_speaker(audio_path)

        assert result is True

    def test_no_speakers_returns_false(self, fixtures_dir: Path) -> None:
        """Audio with no detected speakers returns False."""
        audio_path = _make_audio(fixtures_dir / "none_ms.wav")
        annotation = _make_mock_annotation([])
        mock_pipeline = MagicMock(return_value=annotation)

        with patch("app.pipelines.diarize._load_diarization_pipeline", return_value=mock_pipeline):
            result = detect_multi_speaker(audio_path)

        assert result is False

    def test_returns_bool(self, fixtures_dir: Path) -> None:
        """Function returns a boolean value."""
        audio_path = _make_audio(fixtures_dir / "bool_ms.wav")
        annotation = _make_mock_annotation([(0.0, 1.0, "SPEAKER_00")])
        mock_pipeline = MagicMock(return_value=annotation)

        with patch("app.pipelines.diarize._load_diarization_pipeline", return_value=mock_pipeline):
            result = detect_multi_speaker(audio_path)

        assert isinstance(result, bool)

    def test_file_not_found_raises_error(self, fixtures_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            detect_multi_speaker(fixtures_dir / "nonexistent.wav")


class TestSelectDominantSpeaker:
    def test_single_speaker(self) -> None:
        """Single speaker is selected as dominant."""
        segments = [
            SpeakerSegment(0.0, 5.0, "SPEAKER_00"),
        ]
        assert select_dominant_speaker(segments) == "SPEAKER_00"

    def test_dominant_by_duration(self) -> None:
        """Speaker with most total speech time is selected."""
        segments = [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),   # 2s
            SpeakerSegment(2.5, 7.0, "SPEAKER_01"),   # 4.5s
            SpeakerSegment(7.5, 9.0, "SPEAKER_00"),   # 1.5s → total 3.5s
        ]
        assert select_dominant_speaker(segments) == "SPEAKER_01"

    def test_dominant_across_many_segments(self) -> None:
        """Dominant speaker selected from many small segments."""
        segments = [
            SpeakerSegment(0.0, 1.0, "SPEAKER_00"),   # 1s
            SpeakerSegment(1.0, 2.0, "SPEAKER_01"),   # 1s
            SpeakerSegment(2.0, 3.0, "SPEAKER_00"),   # 1s
            SpeakerSegment(3.0, 4.0, "SPEAKER_01"),   # 1s
            SpeakerSegment(4.0, 6.0, "SPEAKER_00"),   # 2s → total 4s
        ]
        # SPEAKER_00: 4s, SPEAKER_01: 2s
        assert select_dominant_speaker(segments) == "SPEAKER_00"

    def test_three_speakers(self) -> None:
        """Dominant speaker selected from three speakers."""
        segments = [
            SpeakerSegment(0.0, 1.0, "SPEAKER_00"),    # 1s
            SpeakerSegment(1.0, 4.0, "SPEAKER_01"),    # 3s
            SpeakerSegment(4.0, 5.5, "SPEAKER_02"),    # 1.5s
            SpeakerSegment(5.5, 6.0, "SPEAKER_00"),    # 0.5s → total 1.5s
        ]
        assert select_dominant_speaker(segments) == "SPEAKER_01"

    def test_empty_segments_raises_error(self) -> None:
        """Empty segment list raises ValueError."""
        with pytest.raises(ValueError, match="empty segments"):
            select_dominant_speaker([])

    def test_returns_string(self) -> None:
        """Return value is a string."""
        segments = [SpeakerSegment(0.0, 1.0, "SPEAKER_00")]
        result = select_dominant_speaker(segments)
        assert isinstance(result, str)


class TestLoadDiarizationPipeline:
    def test_caches_pipeline(self) -> None:
        """_load_diarization_pipeline caches after first load."""
        import app.pipelines.diarize as diarize_mod

        original = diarize_mod._diarization_pipeline
        try:
            diarize_mod._diarization_pipeline = None

            mock_pipeline = MagicMock()
            mock_from_pretrained = MagicMock(return_value=mock_pipeline)

            with (
                patch("app.pipelines.preprocess._ensure_torchaudio_compat"),
                patch("pyannote.audio.Pipeline.from_pretrained", mock_from_pretrained),
            ):
                result1 = diarize_mod._load_diarization_pipeline()
                result2 = diarize_mod._load_diarization_pipeline()

            assert result1 is result2
            mock_from_pretrained.assert_called_once()
        finally:
            diarize_mod._diarization_pipeline = original
