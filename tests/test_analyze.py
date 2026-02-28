"""Tests for the audio analysis pipeline.

Covers merge_segments, score_segment, select_best_segments,
sub-score helpers, compute_speaker_embedding, and build_speaker_profile.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from app.config import settings
from app.pipelines.analyze import (
    CLIPPING_THRESHOLD,
    MAX_ACCEPTABLE_DURATION,
    MAX_SEGMENT_DURATION,
    MIN_ACCEPTABLE_DURATION,
    ScoredSegment,
    _clipping_score,
    _compute_quality_summary,
    _compute_snr_db,
    _duration_score,
    _snr_proxy_score,
    build_speaker_profile,
    compute_speaker_embedding,
    merge_segments,
    score_segment,
    select_best_segments,
)
from app.pipelines.diarize import SpeakerSegment


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    d = tmp_path / "fixtures"
    d.mkdir()
    return d


def _make_audio(
    path: Path,
    duration: float = 5.0,
    sr: int = 16000,
    amplitude: float = 0.5,
    freq: float = 440.0,
) -> Path:
    """Create a sine-wave WAV file for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    samples = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), samples, sr, subtype="PCM_16")
    return path


# ── merge_segments ──────────────────────────────────────────────────


class TestMergeSegments:
    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert merge_segments([]) == []

    def test_single_segment(self) -> None:
        """Single segment is returned unchanged."""
        segs = [SpeakerSegment(0.0, 5.0, "SPEAKER_00")]
        result = merge_segments(segs)
        assert len(result) == 1
        assert result[0] == segs[0]

    def test_merges_same_speaker(self) -> None:
        """Consecutive segments from the same speaker are merged."""
        segs = [
            SpeakerSegment(0.0, 4.0, "SPEAKER_00"),
            SpeakerSegment(4.0, 7.0, "SPEAKER_00"),
        ]
        result = merge_segments(segs)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 7.0
        assert result[0].speaker_id == "SPEAKER_00"

    def test_no_merge_across_speakers(self) -> None:
        """Segments from different speakers are not merged."""
        segs = [
            SpeakerSegment(0.0, 4.0, "SPEAKER_00"),
            SpeakerSegment(4.0, 8.0, "SPEAKER_01"),
        ]
        result = merge_segments(segs)
        assert len(result) == 2
        assert result[0].speaker_id == "SPEAKER_00"
        assert result[1].speaker_id == "SPEAKER_01"

    def test_respects_max_duration(self) -> None:
        """Merging stops when combined duration exceeds max_duration."""
        segs = [
            SpeakerSegment(0.0, 8.0, "SPEAKER_00"),
            SpeakerSegment(8.0, 16.0, "SPEAKER_00"),
        ]
        result = merge_segments(segs, max_duration=MAX_SEGMENT_DURATION)
        # 8 + 8 = 16 > 15, so they should not merge
        assert len(result) == 2

    def test_merges_within_max_duration(self) -> None:
        """Segments merge when combined duration stays within max."""
        segs = [
            SpeakerSegment(0.0, 5.0, "SPEAKER_00"),
            SpeakerSegment(5.0, 10.0, "SPEAKER_00"),
        ]
        result = merge_segments(segs, max_duration=MAX_SEGMENT_DURATION)
        # 5 + 5 = 10 <= 15
        assert len(result) == 1
        assert result[0].end == 10.0

    def test_unsorted_input(self) -> None:
        """Segments are sorted by start time before merging."""
        segs = [
            SpeakerSegment(4.0, 7.0, "SPEAKER_00"),
            SpeakerSegment(0.0, 4.0, "SPEAKER_00"),
        ]
        result = merge_segments(segs)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 7.0

    def test_alternating_speakers(self) -> None:
        """Alternating speakers produces no merges."""
        segs = [
            SpeakerSegment(0.0, 3.0, "SPEAKER_00"),
            SpeakerSegment(3.0, 6.0, "SPEAKER_01"),
            SpeakerSegment(6.0, 9.0, "SPEAKER_00"),
        ]
        result = merge_segments(segs)
        assert len(result) == 3

    def test_returns_speaker_segments(self) -> None:
        """Results are SpeakerSegment namedtuples."""
        segs = [SpeakerSegment(0.0, 5.0, "SPEAKER_00")]
        result = merge_segments(segs)
        assert isinstance(result[0], SpeakerSegment)

    def test_multiple_merges_chain(self) -> None:
        """Three short same-speaker segments merge into one if within max."""
        segs = [
            SpeakerSegment(0.0, 3.0, "SPEAKER_00"),
            SpeakerSegment(3.0, 6.0, "SPEAKER_00"),
            SpeakerSegment(6.0, 9.0, "SPEAKER_00"),
        ]
        result = merge_segments(segs, max_duration=15.0)
        assert len(result) == 1
        assert result[0].end == 9.0


# ── score_segment ───────────────────────────────────────────────────


class TestScoreSegment:
    def test_basic_scoring(self, fixtures_dir: Path) -> None:
        """Scoring returns a ScoredSegment with a quality_score in [0, 1]."""
        audio_path = _make_audio(fixtures_dir / "basic.wav", duration=5.0)
        seg = SpeakerSegment(0.0, 5.0, "SPEAKER_00")
        result = score_segment(audio_path, seg)

        assert isinstance(result, ScoredSegment)
        assert 0.0 <= result.quality_score <= 1.0

    def test_preserves_segment_info(self, fixtures_dir: Path) -> None:
        """Scored segment preserves start, end, speaker_id."""
        audio_path = _make_audio(fixtures_dir / "info.wav", duration=5.0)
        seg = SpeakerSegment(1.0, 4.0, "SPEAKER_01")
        result = score_segment(audio_path, seg)

        assert result.start == 1.0
        assert result.end == 4.0
        assert result.speaker_id == "SPEAKER_01"

    def test_file_not_found(self, fixtures_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        seg = SpeakerSegment(0.0, 1.0, "SPEAKER_00")
        with pytest.raises(FileNotFoundError):
            score_segment(fixtures_dir / "missing.wav", seg)

    def test_short_segment_lower_duration_score(self, fixtures_dir: Path) -> None:
        """Very short segment gets a lower score than one in the sweet spot."""
        audio_path = _make_audio(fixtures_dir / "short_vs_good.wav", duration=10.0)
        short_seg = SpeakerSegment(0.0, 1.0, "SPEAKER_00")
        good_seg = SpeakerSegment(0.0, 5.0, "SPEAKER_00")

        short_result = score_segment(audio_path, short_seg)
        good_result = score_segment(audio_path, good_seg)

        assert short_result.quality_score < good_result.quality_score

    def test_clipping_reduces_score(self, fixtures_dir: Path) -> None:
        """Audio with clipping gets a lower score than clean audio."""
        clean_path = _make_audio(
            fixtures_dir / "clean.wav", duration=5.0, amplitude=0.5
        )
        clipped_path = fixtures_dir / "clipped.wav"
        sr = 16000
        # Create audio with many clipped samples
        t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False, dtype=np.float32)
        samples = np.ones_like(t) * 0.999  # near-clipping
        sf.write(str(clipped_path), samples, sr, subtype="FLOAT")

        seg = SpeakerSegment(0.0, 5.0, "SPEAKER_00")
        clean_score = score_segment(clean_path, seg)
        clipped_score = score_segment(clipped_path, seg)

        assert clean_score.quality_score > clipped_score.quality_score

    def test_stereo_audio(self, fixtures_dir: Path) -> None:
        """Stereo audio is handled (first channel used)."""
        path = fixtures_dir / "stereo.wav"
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.3 * np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([left, right]).astype(np.float32)
        sf.write(str(path), stereo, sr, subtype="PCM_16")

        seg = SpeakerSegment(0.0, 5.0, "SPEAKER_00")
        result = score_segment(path, seg)
        assert 0.0 <= result.quality_score <= 1.0

    def test_score_is_rounded(self, fixtures_dir: Path) -> None:
        """Quality score is rounded to 4 decimal places."""
        audio_path = _make_audio(fixtures_dir / "rounded.wav", duration=5.0)
        seg = SpeakerSegment(0.0, 5.0, "SPEAKER_00")
        result = score_segment(audio_path, seg)

        score_str = str(result.quality_score)
        if "." in score_str:
            decimals = len(score_str.split(".")[1])
            assert decimals <= 4


# ── select_best_segments ────────────────────────────────────────────


class TestSelectBestSegments:
    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert select_best_segments([]) == []

    def test_selects_highest_quality(self) -> None:
        """Higher-quality segments are selected first."""
        scored = [
            ScoredSegment(0.0, 5.0, "SPEAKER_00", 0.5),
            ScoredSegment(5.0, 10.0, "SPEAKER_00", 0.9),
            ScoredSegment(10.0, 15.0, "SPEAKER_00", 0.3),
        ]
        result = select_best_segments(scored, target_min=5.0, target_max=10.0)

        # Should select the 0.9 segment first, then 0.5 to reach target_min
        assert len(result) >= 1
        scores = {s.quality_score for s in result}
        assert 0.9 in scores

    def test_stops_at_target_min(self) -> None:
        """Selection stops once target_min duration is reached."""
        scored = [
            ScoredSegment(0.0, 6.0, "SPEAKER_00", 0.9),    # 6s
            ScoredSegment(6.0, 12.0, "SPEAKER_00", 0.8),   # 6s → total 12s
            ScoredSegment(12.0, 18.0, "SPEAKER_00", 0.7),  # 6s → total 18s
        ]
        result = select_best_segments(scored, target_min=10.0, target_max=30.0)
        total = sum(s.end - s.start for s in result)
        assert total >= 10.0

    def test_respects_target_max(self) -> None:
        """Does not exceed target_max if already past target_min."""
        scored = [
            ScoredSegment(0.0, 8.0, "SPEAKER_00", 0.9),     # 8s
            ScoredSegment(8.0, 16.0, "SPEAKER_00", 0.8),    # 8s → total 16s
            ScoredSegment(16.0, 24.0, "SPEAKER_00", 0.7),   # 8s → total 24s
        ]
        result = select_best_segments(scored, target_min=10.0, target_max=20.0)
        total = sum(s.end - s.start for s in result)
        # Should have 8+8=16 (>= target_min), stops before adding third (would be 24 > 20)
        assert total <= 20.0

    def test_returns_chronological_order(self) -> None:
        """Selected segments are returned in chronological order."""
        scored = [
            ScoredSegment(10.0, 15.0, "SPEAKER_00", 0.9),
            ScoredSegment(0.0, 5.0, "SPEAKER_00", 0.8),
            ScoredSegment(5.0, 10.0, "SPEAKER_00", 0.7),
        ]
        result = select_best_segments(scored, target_min=10.0, target_max=30.0)
        starts = [s.start for s in result]
        assert starts == sorted(starts)

    def test_all_segments_when_under_target(self) -> None:
        """All segments returned when total is under target_min."""
        scored = [
            ScoredSegment(0.0, 3.0, "SPEAKER_00", 0.9),
            ScoredSegment(3.0, 6.0, "SPEAKER_00", 0.8),
        ]
        # Total = 6s, under target_min=10
        result = select_best_segments(scored, target_min=10.0, target_max=30.0)
        assert len(result) == 2

    def test_includes_past_max_if_under_min(self) -> None:
        """If under target_min, includes a segment even if it pushes past target_max."""
        scored = [
            ScoredSegment(0.0, 8.0, "SPEAKER_00", 0.9),    # 8s
            ScoredSegment(8.0, 20.0, "SPEAKER_00", 0.8),   # 12s → total 20s
        ]
        # target_min=15, target_max=18. After first seg (8s), still under 15, so next is added
        result = select_best_segments(scored, target_min=15.0, target_max=18.0)
        assert len(result) == 2

    def test_skips_segment_when_over_max_and_past_min(self) -> None:
        """Stops adding when past min and next segment exceeds max."""
        scored = [
            ScoredSegment(0.0, 12.0, "SPEAKER_00", 0.9),   # 12s, added → total=12
            ScoredSegment(12.0, 22.0, "SPEAKER_00", 0.8),  # 10s, would push to 22 > 15
        ]
        # target_min=0, so total=0 already >= target_min at first check
        # First seg: total+dur=12 > 5 (target_max), total=0 >= 0 → break
        result = select_best_segments(scored, target_min=0.0, target_max=5.0)
        assert len(result) == 0

    def test_returns_scored_segments(self) -> None:
        """Results are ScoredSegment namedtuples."""
        scored = [ScoredSegment(0.0, 5.0, "SPEAKER_00", 0.9)]
        result = select_best_segments(scored, target_min=1.0, target_max=10.0)
        assert isinstance(result[0], ScoredSegment)


# ── Sub-score helpers ───────────────────────────────────────────────


class TestSnrProxyScore:
    def test_empty_chunk(self) -> None:
        """Empty chunk returns 0.0."""
        assert _snr_proxy_score(np.array([], dtype=np.float32)) == 0.0

    def test_short_chunk(self) -> None:
        """Chunk shorter than one frame returns 0.0."""
        chunk = np.zeros(100, dtype=np.float32)
        assert _snr_proxy_score(chunk) == 0.0

    def test_silence(self) -> None:
        """Silent audio (zero signal energy) returns 0.0."""
        chunk = np.zeros(4096, dtype=np.float32)
        # All-zero signal → signal_energy < 1e-10 → 0.0
        assert _snr_proxy_score(chunk) == 0.0

    def test_pure_tone(self) -> None:
        """Pure tone gets a positive score."""
        sr = 16000
        t = np.linspace(0, 0.5, sr // 2, endpoint=False, dtype=np.float32)
        chunk = 0.5 * np.sin(2 * np.pi * 440 * t)
        score = _snr_proxy_score(chunk.astype(np.float32))
        assert score > 0.0

    def test_score_range(self) -> None:
        """Score is in [0, 1]."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        chunk = 0.5 * np.sin(2 * np.pi * 440 * t)
        score = _snr_proxy_score(chunk.astype(np.float32))
        assert 0.0 <= score <= 1.0

    def test_zero_noise_floor(self) -> None:
        """Very low noise floor returns 1.0."""
        # Create chunk where quietest frames have near-zero energy
        chunk = np.zeros(4096, dtype=np.float32)
        # Fill most frames with signal, leave some near-zero
        chunk[1024:3072] = 0.5
        score = _snr_proxy_score(chunk)
        assert score == 1.0


class TestClippingScore:
    def test_empty_chunk(self) -> None:
        """Empty chunk returns 0.0."""
        assert _clipping_score(np.array([], dtype=np.float32)) == 0.0

    def test_no_clipping(self) -> None:
        """Audio below clipping threshold returns 1.0."""
        chunk = np.full(1000, 0.5, dtype=np.float32)
        assert _clipping_score(chunk) == 1.0

    def test_all_clipped(self) -> None:
        """Fully clipped audio returns 0.0."""
        chunk = np.full(1000, 1.0, dtype=np.float32)
        score = _clipping_score(chunk)
        assert score == 0.0

    def test_partial_clipping(self) -> None:
        """Partially clipped audio returns between 0 and 1."""
        chunk = np.full(1000, 0.5, dtype=np.float32)
        chunk[:50] = 1.0  # 5% clipped
        score = _clipping_score(chunk)
        assert 0.0 < score < 1.0

    def test_threshold_boundary(self) -> None:
        """Samples exactly at threshold are not considered clipped."""
        chunk = np.full(1000, CLIPPING_THRESHOLD, dtype=np.float32)
        score = _clipping_score(chunk)
        assert score == 1.0


class TestDurationScore:
    def test_zero_duration(self) -> None:
        """Zero duration returns 0.0."""
        assert _duration_score(0.0) == 0.0

    def test_negative_duration(self) -> None:
        """Negative duration returns 0.0."""
        assert _duration_score(-1.0) == 0.0

    def test_min_boundary(self) -> None:
        """At minimum acceptable duration, score is 1.0."""
        assert _duration_score(MIN_ACCEPTABLE_DURATION) == 1.0

    def test_max_boundary(self) -> None:
        """At maximum acceptable duration, score is 1.0."""
        assert _duration_score(MAX_ACCEPTABLE_DURATION) == 1.0

    def test_in_range(self) -> None:
        """Duration in acceptable range returns 1.0."""
        assert _duration_score(10.0) == 1.0

    def test_below_range(self) -> None:
        """Duration below range returns proportional score."""
        score = _duration_score(1.5)
        expected = 1.5 / MIN_ACCEPTABLE_DURATION
        assert score == pytest.approx(expected)

    def test_above_range(self) -> None:
        """Duration above range returns decreasing score."""
        duration = MAX_ACCEPTABLE_DURATION + 10.0
        score = _duration_score(duration)
        assert 0.0 < score < 1.0

    def test_very_long_returns_zero(self) -> None:
        """Very long duration eventually returns 0.0."""
        score = _duration_score(MAX_ACCEPTABLE_DURATION * 2.0)
        assert score == 0.0


# ── _compute_snr_db ─────────────────────────────────────────────────


class TestComputeSnrDb:
    def test_empty_chunk(self) -> None:
        """Empty chunk returns 0.0."""
        assert _compute_snr_db(np.array([], dtype=np.float32)) == 0.0

    def test_short_chunk(self) -> None:
        """Chunk shorter than one frame returns 0.0."""
        assert _compute_snr_db(np.zeros(100, dtype=np.float32)) == 0.0

    def test_silence(self) -> None:
        """Silent audio returns 0.0 dB."""
        assert _compute_snr_db(np.zeros(4096, dtype=np.float32)) == 0.0

    def test_pure_tone_positive_db(self) -> None:
        """Pure tone returns a positive dB value."""
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        chunk = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        assert _compute_snr_db(chunk) > 0.0


# ── _compute_quality_summary ────────────────────────────────────────


class TestComputeQualitySummary:
    def test_clean_segments(self, fixtures_dir: Path) -> None:
        """Clean audio produces zero clipped segments."""
        audio_path = _make_audio(fixtures_dir / "clean_q.wav", duration=10.0)
        data, sr = sf.read(str(audio_path), dtype="float32")
        segments = [ScoredSegment(0.0, 5.0, "SPEAKER_00", 0.9)]
        summary = _compute_quality_summary(data, sr, segments)
        assert "mean_snr" in summary
        assert summary["clipped_segments"] == 0

    def test_clipped_segment_counted(self, fixtures_dir: Path) -> None:
        """Segments with clipping are counted."""
        path = fixtures_dir / "clipped_q.wav"
        sr = 16000
        samples = np.ones(int(sr * 5.0), dtype=np.float32)
        sf.write(str(path), samples, sr, subtype="FLOAT")
        data, _ = sf.read(str(path), dtype="float32")
        segments = [ScoredSegment(0.0, 5.0, "SPEAKER_00", 0.5)]
        summary = _compute_quality_summary(data, sr, segments)
        assert summary["clipped_segments"] == 1

    def test_empty_segments(self) -> None:
        """No segments produces zero mean_snr and zero clipped."""
        data = np.zeros(16000, dtype=np.float32)
        summary = _compute_quality_summary(data, 16000, [])
        assert summary["mean_snr"] == 0.0
        assert summary["clipped_segments"] == 0


# ── compute_speaker_embedding ───────────────────────────────────────


class TestComputeSpeakerEmbedding:
    def test_returns_192_dim_array(self, fixtures_dir: Path) -> None:
        """Returns a 192-dimensional numpy array."""
        audio_path = _make_audio(fixtures_dir / "embed.wav", duration=10.0)
        segments = [ScoredSegment(0.0, 5.0, "SPEAKER_00", 0.9)]

        mock_model = MagicMock()
        mock_model.encode_batch.return_value = torch.zeros(1, 1, 192)

        with patch("app.pipelines.analyze._load_ecapa_model", return_value=mock_model):
            embedding = compute_speaker_embedding(audio_path, segments)

        assert embedding.shape == (192,)

    def test_file_not_found(self, fixtures_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        segments = [ScoredSegment(0.0, 1.0, "SPEAKER_00", 0.9)]
        with pytest.raises(FileNotFoundError):
            compute_speaker_embedding(fixtures_dir / "missing.wav", segments)

    def test_resamples_non_16khz(self, fixtures_dir: Path) -> None:
        """Audio at non-16kHz sample rate is resampled."""
        audio_path = _make_audio(fixtures_dir / "44k.wav", duration=5.0, sr=44100)
        segments = [ScoredSegment(0.0, 5.0, "SPEAKER_00", 0.9)]

        mock_model = MagicMock()
        mock_model.encode_batch.return_value = torch.zeros(1, 1, 192)

        with patch("app.pipelines.analyze._load_ecapa_model", return_value=mock_model):
            embedding = compute_speaker_embedding(audio_path, segments)

        assert embedding.shape == (192,)

    def test_stereo_uses_first_channel(self, fixtures_dir: Path) -> None:
        """Stereo audio uses the first channel only."""
        path = fixtures_dir / "stereo_embed.wav"
        sr = 16000
        t = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False, dtype=np.float32)
        stereo = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.3 * np.sin(2 * np.pi * 880 * t),
        ]).astype(np.float32)
        sf.write(str(path), stereo, sr, subtype="PCM_16")

        segments = [ScoredSegment(0.0, 5.0, "SPEAKER_00", 0.9)]
        mock_model = MagicMock()
        mock_model.encode_batch.return_value = torch.zeros(1, 1, 192)

        with patch("app.pipelines.analyze._load_ecapa_model", return_value=mock_model):
            embedding = compute_speaker_embedding(path, segments)

        assert embedding.shape == (192,)


# ── _load_ecapa_model ───────────────────────────────────────────────


class TestLoadEcapaModel:
    def test_caches_model(self) -> None:
        """Model is loaded once and cached."""
        from app.pipelines.preprocess import _ensure_torchaudio_compat

        _ensure_torchaudio_compat()

        import app.pipelines.analyze as analyze_mod

        original = analyze_mod._ecapa_model
        try:
            analyze_mod._ecapa_model = None
            mock_classifier = MagicMock()

            with patch(
                "speechbrain.inference.speaker.EncoderClassifier.from_hparams",
                return_value=mock_classifier,
            ):
                result1 = analyze_mod._load_ecapa_model()
                result2 = analyze_mod._load_ecapa_model()

            assert result1 is result2
        finally:
            analyze_mod._ecapa_model = original


# ── build_speaker_profile ───────────────────────────────────────────


def _profile_context(audio_path: Path, storage: Path) -> ExitStack:
    """Build an ExitStack that mocks all pipeline functions for profile tests."""
    mock_ecapa = MagicMock()
    mock_ecapa.encode_batch.return_value = torch.zeros(1, 1, 192)

    stack = ExitStack()
    stack.enter_context(patch(
        "app.pipelines.preprocess.separate_vocals", return_value=audio_path,
    ))
    stack.enter_context(patch(
        "app.pipelines.preprocess.enhance_speech", return_value=audio_path,
    ))
    stack.enter_context(patch(
        "app.pipelines.diarize.diarize_speakers",
        return_value=[
            SpeakerSegment(0.0, 5.0, "SPEAKER_00"),
            SpeakerSegment(5.0, 10.0, "SPEAKER_01"),
            SpeakerSegment(10.0, 20.0, "SPEAKER_00"),
        ],
    ))
    stack.enter_context(patch(
        "app.pipelines.diarize.select_dominant_speaker",
        return_value="SPEAKER_00",
    ))
    stack.enter_context(patch(
        "app.pipelines.analyze._load_ecapa_model", return_value=mock_ecapa,
    ))
    stack.enter_context(patch.object(settings, "storage_path", storage))
    return stack


class TestBuildSpeakerProfile:
    def test_creates_profile_json(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Profile JSON file is created with expected keys."""
        audio_path = _make_audio(fixtures_dir / "profile.wav", duration=20.0)
        storage = tmp_path / "storage"

        with _profile_context(audio_path, storage):
            profile = build_speaker_profile(audio_path, "job-1")

        assert (storage / "job-1" / "speaker_profile.json").exists()
        with open(storage / "job-1" / "speaker_profile.json") as f:
            saved = json.load(f)
        assert saved["id"] == profile["id"]

    def test_profile_has_required_keys(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Returned profile dict contains all required schema keys."""
        audio_path = _make_audio(fixtures_dir / "keys.wav", duration=20.0)
        storage = tmp_path / "storage"

        with _profile_context(audio_path, storage):
            profile = build_speaker_profile(audio_path, "job-keys")

        required_keys = {
            "id", "created_at", "source_file", "embedding_path",
            "segments", "total_duration_s", "speaker_count",
            "dominant_speaker_id", "quality_summary",
        }
        assert required_keys.issubset(profile.keys())

    def test_embedding_saved(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Embedding .npy file is saved to disk."""
        audio_path = _make_audio(fixtures_dir / "emb_save.wav", duration=20.0)
        storage = tmp_path / "storage"

        with _profile_context(audio_path, storage):
            profile = build_speaker_profile(audio_path, "job-emb")

        embedding_path = Path(profile["embedding_path"])
        assert embedding_path.exists()
        data = np.load(str(embedding_path))
        assert data.shape == (192,)

    def test_segment_wavs_created(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Individual segment WAV files are created."""
        audio_path = _make_audio(fixtures_dir / "seg_wav.wav", duration=20.0)
        storage = tmp_path / "storage"

        with _profile_context(audio_path, storage):
            profile = build_speaker_profile(audio_path, "job-seg")

        for seg_info in profile["segments"]:
            assert Path(seg_info["path"]).exists()

    def test_dominant_speaker_selected(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Profile uses the dominant speaker."""
        audio_path = _make_audio(fixtures_dir / "dominant.wav", duration=20.0)
        storage = tmp_path / "storage"

        with _profile_context(audio_path, storage):
            profile = build_speaker_profile(audio_path, "job-dom")

        assert profile["dominant_speaker_id"] == "SPEAKER_00"
        assert profile["speaker_count"] == 2

    def test_quality_summary_structure(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Quality summary has mean_snr and clipped_segments."""
        audio_path = _make_audio(fixtures_dir / "qual.wav", duration=20.0)
        storage = tmp_path / "storage"

        with _profile_context(audio_path, storage):
            profile = build_speaker_profile(audio_path, "job-qual")

        qs = profile["quality_summary"]
        assert "mean_snr" in qs
        assert "clipped_segments" in qs
        assert isinstance(qs["mean_snr"], float)
        assert isinstance(qs["clipped_segments"], int)

    def test_file_not_found(self, fixtures_dir: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            build_speaker_profile(fixtures_dir / "missing.wav", "job-x")

    def test_source_file_is_filename(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """source_file contains just the filename, not the full path."""
        audio_path = _make_audio(fixtures_dir / "source.wav", duration=20.0)
        storage = tmp_path / "storage"

        with _profile_context(audio_path, storage):
            profile = build_speaker_profile(audio_path, "job-src")

        assert profile["source_file"] == "source.wav"

    def test_stereo_audio_handled(self, fixtures_dir: Path, tmp_path: Path) -> None:
        """Stereo audio is converted to mono during profile building."""
        path = fixtures_dir / "stereo_profile.wav"
        sr = 16000
        t = np.linspace(0, 20.0, int(sr * 20.0), endpoint=False, dtype=np.float32)
        stereo = np.column_stack([
            0.5 * np.sin(2 * np.pi * 440 * t),
            0.3 * np.sin(2 * np.pi * 880 * t),
        ]).astype(np.float32)
        sf.write(str(path), stereo, sr, subtype="PCM_16")

        storage = tmp_path / "storage"

        with _profile_context(path, storage):
            profile = build_speaker_profile(path, "job-stereo")

        assert profile["dominant_speaker_id"] == "SPEAKER_00"
        assert len(profile["segments"]) > 0
