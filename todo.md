# TODO: Korean Voice Cloning TTS — Execution Checklist

---

## Milestone 0: Project Scaffolding

### 0.0 — Set up Python 3.13 virtualenv with uv
- [x] Install `uv` if not already present (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`)
- [x] Create virtualenv with Python 3.13: `uv venv --python 3.13`
- [x] Activate the virtualenv: `source .venv/bin/activate`
- [x] Verify Python version: `python --version` shows 3.13.x
- [x] Verify uv is managing the environment: `uv pip list` works

**Pass criteria:**
- `.venv/` directory exists at project root
- `python --version` inside the venv outputs `Python 3.13.x`
- `uv pip list` runs successfully and shows an empty or minimal package list
- `.venv/` is listed in `.gitignore`

---

### 0.1 — Initialize repository and project config
- [x] Create `pyproject.toml` with all dependencies listed in plan Section 11
  - Pin `requires-python = ">=3.13"`
  - Pin `fastapi==0.134.0`
  - Use `uv` as the package manager (configure `[tool.uv]` section if needed)
- [x] Configure linting (ruff), formatting (black), type-checking (ty)
- [x] Add `.gitignore` for Python, `.venv/`, audio files, model weights, `.env`, Docker volumes
- [x] Initialize git repo and make initial commit
- [x] Install all dependencies via `uv sync` or `uv pip install -e .`

**Pass criteria:**
- `git status` shows clean working tree after initial commit
- `pyproject.toml` specifies `requires-python = ">=3.13"` and `fastapi==0.134.0`
- `uv sync` succeeds and all dependencies are installed into `.venv/`
- `python -c "import fastapi; print(fastapi.__version__)"` outputs `0.134.0`
- Linting config is present and runnable

---

### 0.2 — Create directory structure
- [x] Create full directory tree matching plan Section 5:
  ```
  app/__init__.py
  app/main.py
  app/config.py
  app/api/routes.py
  app/api/schemas.py
  app/pipelines/ingest.py
  app/pipelines/preprocess.py
  app/pipelines/diarize.py
  app/pipelines/analyze.py
  app/pipelines/korean_text.py
  app/services/tts_backend.py
  app/services/synthesis.py
  app/services/postprocess.py
  app/models/db.py
  app/workers/tasks.py
  tests/test_ingest.py
  tests/test_preprocess.py
  tests/test_korean_text.py
  tests/test_tts_backend.py
  tests/test_api.py
  tests/fixtures/
  scripts/e2e_smoke.sh
  scripts/benchmark.py
  docker/Dockerfile
  docker/docker-compose.yml
  ```
- [x] Each Python file has a module docstring and placeholder imports

**Pass criteria:**
- All files exist at the correct paths
- `python -c "import app"` does not error
- Directory structure matches plan Section 5 exactly

---

### 0.3 — Set up configuration management
- [x] Implement `app/config.py` using `pydantic-settings`
- [x] Define settings: database URL, Redis URL, storage path, model paths, FFmpeg path, device (cpu/cuda/mps), debug flag
- [x] Support `.env` file loading and environment variable overrides

**Pass criteria:**
- `from app.config import settings` works and returns typed config object
- Missing required env vars raise a clear validation error
- `.env.example` file documents all settings with defaults

---

## Milestone 1: Working Skeleton (Weeks 1-2)

### 1.1 — Docker Compose stack
- [x] Write `docker/Dockerfile` for the FastAPI app (Python 3.13, uv for dependency management, FFmpeg installed)
- [x] Write `docker/docker-compose.yml` with services: `api`, `worker`, `redis`, `postgres`
- [x] Add health check endpoints for each service
- [x] Configure volume mounts for local file storage and model cache

**Pass criteria:**
- `docker compose up` boots all 4 services without errors
- `curl http://localhost:8000/health` returns `{"status": "ok"}`
- Redis is reachable from both `api` and `worker` containers
- Postgres is reachable and migrations run on startup

---

### 1.2 — Database models and migrations
- [x] Implement `app/models/db.py` with SQLAlchemy 2.0 models:
  - `Job` table: id, status (pending/processing/completed/failed), stage, progress, error_message, created_at, updated_at, input_file_path, speaker_profile_id
  - `SpeakerProfile` table: id, name, embedding_path, segments_json, metadata_json, created_at
  - `AudioOutput` table: id, job_id (nullable), speaker_profile_id, text_input, file_path, format, created_at
- [x] Set up Alembic for migrations (or use `create_all` for V1)

**Pass criteria:**
- Tables are created in Postgres on startup
- Models can be imported and instantiated without error
- Foreign key relationships are correctly defined

---

### 1.3 — FastAPI app skeleton with health and upload endpoints
- [x] Implement `app/main.py`: create FastAPI app, include router, configure CORS, lifespan events
- [x] Implement `app/api/schemas.py`: Pydantic models for all request/response shapes defined in plan Section 3 API Contract
- [x] Implement `app/api/routes.py` with endpoints:
  - `POST /v1/voices/clone` — accept multipart file upload, validate file type (video/audio MIME types), create Job record, persist file to storage, enqueue Celery task, return `{job_id, status}`
  - `GET /v1/jobs/{job_id}` — return job status, stage, progress, errors, speaker_profile_id
  - `POST /v1/tts/synthesize` — stub (returns 501 for now)
  - `GET /v1/audio/{audio_id}` — stub (returns 501 for now)
  - `GET /v1/voices` — list speaker profiles
  - `DELETE /v1/voices/{id}` — delete speaker profile + associated files

**Pass criteria:**
- Upload a `.wav` file via `curl -F "file=@test.wav" http://localhost:8000/v1/voices/clone` and receive `{job_id, status: "pending"}`
- Uploaded file is persisted to configured storage path
- Job record exists in Postgres with correct initial state
- `GET /v1/jobs/{job_id}` returns the job status
- Invalid file types (e.g., `.txt`) are rejected with 422
- `tests/test_api.py` passes with at least 3 test cases (upload success, upload invalid type, get job status)

---

### 1.4 — Celery worker setup
- [x] Implement `app/workers/tasks.py` with Celery app configuration
- [x] Define `process_voice_clone` task that:
  1. Updates job status to "processing"
  2. Calls ingest pipeline (Task 1.5)
  3. Updates job status to "completed" or "failed" with error details
- [x] Configure task routing, retries, and timeouts

**Pass criteria:**
- Worker connects to Redis broker and starts without errors
- Enqueuing a task from the API results in the worker picking it up
- Job status transitions from "pending" → "processing" → "completed"/"failed"
- Failed tasks record the error message in the Job record

---

### 1.5 — FFmpeg audio extraction and normalization
- [x] Implement `app/pipelines/ingest.py`:
  - `extract_audio(input_path: Path) -> Path` — use FFmpeg to extract audio from any video/audio format → mono WAV, 24kHz, 16-bit PCM
  - Handle edge cases: already-WAV input, stereo→mono downmix, sample rate conversion
  - Validate output: file exists, non-zero size, correct format
- [x] Add FFmpeg subprocess error handling (missing codec, corrupt file, etc.)

**Pass criteria:**
- MP4 video → mono WAV 24kHz 16-bit output
- MP3 audio → mono WAV 24kHz 16-bit output
- WAV stereo → mono WAV 24kHz 16-bit output
- Corrupt file input → raises typed error with descriptive message
- Output file passes `soundfile.info()` validation (correct channels, sample rate, subtype)
- `tests/test_ingest.py` passes with test cases for each format

---

### 1.6 — Silero VAD segmentation
- [x] Implement VAD in `app/pipelines/preprocess.py`:
  - `detect_speech_segments(audio_path: Path) -> list[Segment]` where `Segment = NamedTuple(start: float, end: float, confidence: float)`
  - Use Silero VAD with threshold=0.5, min_speech_duration=250ms
  - Merge adjacent segments with gaps < 300ms
  - Filter segments shorter than min_speech_duration
- [x] Handle edge case: no speech detected → return empty list with warning

**Pass criteria:**
- On a known test audio file with speech + silence, returns correct segment boundaries (within 100ms tolerance)
- Silent audio file → returns empty list
- Continuous speech → returns single segment spanning full duration
- Segment timestamps are in seconds (float)
- `tests/test_preprocess.py` passes with at least 3 test cases

---

### 1.7 — CI pipeline
- [x] Set up GitHub Actions (or local pre-commit hooks) for:
  - Linting (ruff)
  - Type checking (ty)
  - Unit tests (pytest)
- [x] Ensure CI runs on every push/PR
- [x] Added pytest-cov with 98% minimum coverage requirement

**Pass criteria:**
- `ruff check .` passes with zero errors
- `ty app/` passes (or has explicit ignores documented)
- `pytest tests/` passes all tests
- CI pipeline runs and reports pass/fail status
- Test coverage >= 98% (currently 100%)

---

## Milestone 2: Reference Analysis Pipeline (Weeks 3-4)

### 2.1 — Background noise detection
- [x] Implement `detect_needs_separation(audio_path: Path) -> bool` in `app/pipelines/preprocess.py`
- [x] Use SNR estimation or spectral analysis to determine if background music/noise is significant
- [x] Threshold: if estimated SNR < 20dB, flag for separation

**Pass criteria:**
- Clean speech recording → returns `False`
- Speech over music → returns `True`
- Speech with ambient noise → returns `True`
- Pure music (no speech) → returns `True`

---

### 2.2 — Demucs source separation
- [x] Implement `separate_vocals(audio_path: Path) -> Path` in `app/pipelines/preprocess.py`
  - Use `htdemucs` model for vocal/music separation
  - Return path to isolated vocals track
  - Conditional: skip if `detect_needs_separation` returns False
- [x] Handle MPS/CUDA/CPU device selection from config
- [x] Manage model loading (lazy load, cache in memory)

**Pass criteria:**
- Audio with background music → vocals track with music significantly reduced
- Clean speech → passes through unchanged (skips separation)
- Output is valid WAV file at same sample rate
- Memory usage stays within plan VRAM budget (4-6 GB peak)

---

### 2.3 — DeepFilterNet speech enhancement
- [x] Implement `enhance_speech(audio_path: Path) -> Path` in `app/pipelines/preprocess.py`
  - Apply DeepFilterNet for residual noise removal
  - Preserve speech quality (no artifacts on clean speech)
- [x] Run after Demucs (if used) or directly on input (if clean)

**Pass criteria:**
- Noisy speech → cleaner output with improved SNR
- Clean speech → output quality not degraded (no added artifacts)
- Output is valid WAV, same sample rate and bit depth

---

### 2.4 — Speaker diarization
- [x] Implement `app/pipelines/diarize.py`:
  - `diarize_speakers(audio_path: Path) -> list[SpeakerSegment]` where `SpeakerSegment` has start, end, speaker_id
  - Use pyannote.audio 3.1
  - Conditional: only run on audio with multiple detected speakers
- [x] Implement `detect_multi_speaker(audio_path: Path) -> bool` — quick check for speaker count
- [x] Implement `select_dominant_speaker(segments: list[SpeakerSegment]) -> str` — pick speaker with most total speech time

**Pass criteria:**
- Single-speaker audio → returns all segments with same speaker_id
- Two-speaker audio → correctly separates into two speaker_ids
- Dominant speaker selection picks the speaker with the most total speech duration
- On internal test set, dominant speaker selected correctly >= 90% of the time

---

### 2.5 — Segment merging and quality scoring
- [x] Implement segment merging in `app/pipelines/analyze.py`:
  - Merge speaker segments into clips of 3-15 seconds, splitting at silence gaps
  - Respect speaker boundaries (don't merge across speakers)
- [x] Implement quality scoring:
  - SNR proxy estimation per segment
  - Clipping detection (peak amplitude > 0.99)
  - Duration check (3-30s acceptable range)
  - Composite quality score (0.0 - 1.0)
- [x] Rank segments by quality score, select top N (enough for 10-30s total reference)

**Pass criteria:**
- Segments are between 3-15 seconds long
- Clipped segments receive lower quality scores
- Higher SNR segments receive higher quality scores
- Selected segments sum to 10-30 seconds total duration
- Deterministic: same input always produces same ranking

---

### 2.6 — Speaker profile builder
- [x] Implement `build_speaker_profile(audio_path: Path, job_id: str) -> SpeakerProfile` in `app/pipelines/analyze.py`
  - Orchestrate: noise detection → separation → enhancement → VAD → diarization → segment scoring → profile assembly
  - Generate speaker embedding (ECAPA-TDNN via SpeechBrain)
  - Save profile artifact: `speaker_profile.json` with schema:
    ```json
    {
      "id": "uuid",
      "created_at": "ISO8601",
      "source_file": "original_filename",
      "embedding_path": "path/to/embedding.npy",
      "segments": [
        {"path": "path/to/segment_001.wav", "start": 0.0, "end": 5.2, "quality_score": 0.85}
      ],
      "total_duration_s": 15.6,
      "speaker_count": 2,
      "dominant_speaker_id": "SPEAKER_00",
      "quality_summary": {"mean_snr": 25.3, "clipped_segments": 0}
    }
    ```
- [x] Store profile in database (SpeakerProfile table)
- [x] Wire into Celery task from 1.4

**Pass criteria:**
- Upload video with one speaker → speaker profile created with valid embedding and segments
- Upload video with two speakers → dominant speaker correctly selected, only their segments in profile
- Profile is deterministic for same input
- `speaker_profile.json` validates against defined schema
- Profile segments total between 10-30 seconds of audio
- Database record created with correct foreign keys

---

### 2.7 — EBU R128 loudness normalization
- [x] Implement `normalize_loudness(audio_path: Path, target_lufs: float = -23.0) -> Path` in `app/pipelines/preprocess.py`
  - Use `pyloudnorm` for EBU R128 loudness measurement and normalization
  - Apply to reference segments before building speaker profile

**Pass criteria:**
- Input at -30 LUFS → output at -23 LUFS (within 0.5 LUFS tolerance)
- Input at -18 LUFS → output at -23 LUFS (within 0.5 LUFS tolerance)
- No clipping introduced by normalization
- Output is valid WAV

---

### 2.8 — Error handling for analysis pipeline
- [ ] Define typed error classes:
  - `NoSpeechDetectedError` — VAD found no speech
  - `InsufficientAudioError` — total speech < 3 seconds
  - `AudioCorruptError` — file cannot be decoded
  - `ProcessingError` — generic pipeline failure with stage info
- [ ] Map all errors to API error responses with actionable user messages
- [ ] Wire error codes into Job record on failure

**Pass criteria:**
- Silent audio → Job fails with `NoSpeechDetectedError`, message: "No speech detected in the uploaded file"
- 1-second audio clip → Job fails with `InsufficientAudioError`, message: "At least 3 seconds of speech required"
- Corrupt file → Job fails with `AudioCorruptError`, descriptive message
- `GET /v1/jobs/{job_id}` returns error code and user-friendly message

---

## Milestone 3: End-to-End TTS (Weeks 5-7)

### 3.1 — TTSBackend protocol and Fish Speech integration
- [ ] Implement `app/services/tts_backend.py`:
  - Define `TTSBackend` Protocol with methods:
    - `load_speaker_profile(reference_audio: Path) -> SpeakerProfile`
    - `synthesize(text: str, speaker: SpeakerProfile, **kwargs) -> AudioSegment`
  - Define `SpeakerProfile` and `AudioSegment` data classes
- [ ] Implement `FishSpeechBackend(TTSBackend)`:
  - Install/configure Fish Speech (from source or package)
  - Implement `load_speaker_profile` using Fish Speech's reference audio loading
  - Implement `synthesize` using Fish Speech's inference API
  - Handle MPS/CUDA/CPU device selection
  - Lazy model loading with configurable model path

**Pass criteria:**
- `FishSpeechBackend` instantiates without error on target hardware
- Given a reference audio file and Korean text, produces audio output
- Output is valid WAV/PCM audio data
- Backend is selectable via config (e.g., `TTS_BACKEND=fish_speech`)
- `tests/test_tts_backend.py` has at least 2 tests: init and basic synthesis

---

### 3.2 — Korean text normalization
- [ ] Implement `app/pipelines/korean_text.py`:
  - `normalize_korean_text(text: str) -> str` with the following transformations:
    1. Unicode NFC normalization
    2. Number verbalization with context awareness:
       - Sino-Korean: `3개 → 세 개`, `5시 → 다섯 시`
       - Counter-based: `3명 → 세 명`, `5번 → 오 번`
       - Years: `2024년 → 이천이십사년`
       - Months/days: `3월 → 삼월`, `15일 → 십오일`
    3. Currency: `₩50,000 → 오만 원`, `$100 → 백 달러`
    4. Dates: `2024년 3월 15일 → 이천이십사년 삼월 십오일`
    5. English abbreviations: `AI → 에이아이`, `CEO → 씨이오`
    6. Phone numbers: `010-1234-5678 → 공일공 일이삼사 오육칠팔`

**Pass criteria:**
- All example transformations from the plan produce correct output
- Mixed Korean-English text handled without errors
- `3개` → `세 개` (native Korean counter)
- `₩50,000` → `오만 원`
- `AI` → `에이아이`
- Edge case: empty string → empty string
- Edge case: pure Korean text with no numbers → passes through unchanged
- `tests/test_korean_text.py` passes with >= 20 test cases covering all rule categories

---

### 3.3 — G2P and morphological analysis integration
- [ ] Integrate Mecab-ko for morphological analysis:
  - Word segmentation
  - POS tagging for disambiguation
- [ ] Integrate g2pK for grapheme-to-phoneme conversion:
  - All rules: coda neutralization → liaison → nasalization → lateralization → palatalization → aspiration → tensification → morphophonemic
- [ ] Implement `text_to_phonemes(text: str) -> str` pipeline:
  - Normalize → Mecab analysis → g2pK conversion → phoneme sequence
- [ ] Create custom lexicon override system for g2pK failures

**Pass criteria:**
- `밖` → `[박]` (coda neutralization)
- `음악을` → `[으마글]` (liaison)
- `학문` → `[항문]` (nasalization)
- `신라` → `[실라]` (lateralization)
- `같이` → `[가치]` (palatalization)
- `좋다` → `[조타]` (aspiration)
- Custom lexicon overrides work (add word, verify output changes)
- Pipeline handles mixed Korean-English input
- `tests/test_korean_text.py` expanded with G2P test cases

---

### 3.4 — Korean regression test sentence set
- [ ] Create curated test set covering:
  - All phonological rules (min 2 examples each)
  - Native Korean vs Sino-Korean number systems
  - Mixed Korean-English text (brand names, tech terms)
  - Honorific level variations (해요체, 합쇼체, 해체)
  - Long compound words
  - Abbreviations and acronyms
- [ ] Store in `tests/fixtures/korean_regression_sentences.json` with format:
  ```json
  [
    {"input": "raw text", "expected_normalized": "normalized text", "expected_phonemes": "phoneme sequence", "rule_tags": ["nasalization", "number"]}
  ]
  ```
- [ ] Integrate into test suite as parametrized pytest tests

**Pass criteria:**
- At least 50 regression sentences covering all rule categories
- All sentences pass normalization and G2P conversion
- Zero regressions when run: every sentence produces expected output
- Test set is version-controlled and documented

---

### 3.5 — Synthesis orchestration service
- [ ] Implement `app/services/synthesis.py`:
  - `synthesize_speech(speaker_profile_id: str, text: str) -> AudioOutput`
  - Pipeline: load profile → normalize text → g2pK → TTS backend synthesis → post-process → save output → create DB record
  - Handle long text: split into sentences, synthesize individually, concatenate
  - Sentence splitting aware of Korean punctuation (。, !, ?, .)
- [ ] Wire into API endpoint `POST /v1/tts/synthesize`:
  - Accept `{speaker_profile_id, text, language: "ko"}`
  - Validate speaker_profile_id exists
  - Return `{audio_id, download_url}`
- [ ] Implement `GET /v1/audio/{audio_id}` — serve audio file with correct MIME type

**Pass criteria:**
- `POST /v1/tts/synthesize` with valid profile and Korean text → returns audio_id
- `GET /v1/audio/{audio_id}` → returns playable audio file
- Long text (> 200 chars) is split and synthesized correctly
- Invalid speaker_profile_id → 404 error
- Empty text → 422 validation error
- Database record created with correct metadata

---

### 3.6 — Post-processing pipeline
- [ ] Implement `app/services/postprocess.py`:
  - Loudness normalization of output audio (EBU R128, -23 LUFS)
  - Format conversion: WAV (default) and MP3 output support
  - Trim silence from start/end of output
- [ ] Add format parameter to synthesis endpoint (`format: "wav" | "mp3"`)

**Pass criteria:**
- Output audio is normalized to -23 LUFS (within 0.5 LUFS)
- WAV output is valid and playable
- MP3 output is valid and playable (when requested)
- Leading/trailing silence trimmed (> 500ms silence removed)

---

### 3.7 — Objective metrics pipeline
- [ ] Implement `scripts/benchmark.py`:
  - **CER (Character Error Rate)**: synthesize → back-transcribe with faster-whisper → compare with input text
  - **Speaker similarity**: compute ECAPA-TDNN embeddings for reference and output → cosine similarity
  - **F0 correlation**: extract F0 contours for reference and output → Pearson correlation
  - **DNSMOS**: compute DNS MOS score for audio quality
  - **Latency**: measure time-to-audio for synthesis
- [ ] Output results as JSON and markdown table
- [ ] Define target thresholds from plan Section 8

**Pass criteria:**
- Script runs end-to-end on a test input pair (reference + text)
- CER computation produces valid percentage (0-100%)
- Speaker similarity produces cosine similarity in [0, 1]
- F0 correlation produces Pearson r in [-1, 1]
- DNSMOS produces score in [1, 5]
- Latency measurement is in seconds
- Output includes both JSON and human-readable markdown table
- Target thresholds documented: CER < 10%, similarity > 0.75, F0 > 0.6, DNSMOS > 3.5, latency < 8s

---

### 3.8 — End-to-end integration test
- [ ] Create `scripts/e2e_smoke.sh` that:
  1. Uploads a test audio file via API
  2. Polls job status until complete
  3. Calls synthesis with the created speaker profile and Korean test text
  4. Downloads the output audio
  5. Validates output file is playable and non-empty
- [ ] Run against Docker Compose stack

**Pass criteria:**
- Script completes without errors on at least 3 test inputs
- End-to-end succeeds on curated Korean test set >= 95%
- p95 synthesis latency < 12s for <= 120 characters on target hardware
- Output audio is playable and contains intelligible Korean speech

---

## Milestone 4: Hardening + Model Bake-Off (Weeks 8-10)

### 4.1 — CosyVoice integration
- [ ] Implement `CosyVoiceBackend(TTSBackend)` in `app/services/tts_backend.py`:
  - Install/configure CosyVoice
  - Implement same interface as FishSpeechBackend
  - Handle device selection
- [ ] Verify switchable via config (`TTS_BACKEND=cosyvoice`)

**Pass criteria:**
- CosyVoice backend instantiates and synthesizes Korean speech
- Switching between Fish Speech and CosyVoice is config-only (no code changes)
- Same test inputs produce valid audio from both backends

---

### 4.2 — Model bake-off benchmark
- [ ] Run `scripts/benchmark.py` on both backends with fixed Korean test suite
- [ ] Metrics to compare:
  - CER (intelligibility)
  - Speaker similarity (cosine)
  - F0 correlation (prosody)
  - DNSMOS (audio quality)
  - Synthesis latency
  - Peak VRAM usage
- [ ] Generate side-by-side comparison report (markdown)
- [ ] Include qualitative notes on Korean naturalness

**Pass criteria:**
- Both backends evaluated on identical test set (>= 20 sentences)
- Report includes side-by-side metric comparison table
- Report includes latency/memory tradeoffs
- Report includes qualitative notes on Korean naturalness
- Production-default model chosen and documented with justification

---

### 4.3 — Quality gate for output rejection
- [ ] Implement auto-rejection logic in synthesis pipeline:
  - Reject if DNSMOS < 2.5
  - Reject if back-transcription CER > 30%
  - Reject if output duration is unreasonable (< 0.5s or > 5x expected)
- [ ] Return quality diagnostics in API response
- [ ] Log quality metrics for monitoring

**Pass criteria:**
- Garbage output (random noise) is rejected with diagnostic message
- Good output passes quality gate
- Quality metrics included in API response for synthesized audio
- Rejections logged with details for debugging

---

### 4.4 — Pipeline monitoring and reliability
- [ ] Add structured logging throughout pipeline stages
- [ ] Track and log:
  - Pipeline success rate per stage
  - Latency per stage
  - Error classification breakdown
- [ ] Map every failure to a known error class (no silent failures)

**Pass criteria:**
- Every pipeline stage logs start/end timestamps and outcome
- Error classification covers all observed failure modes
- No unhandled exceptions escape to generic 500 errors
- Success rate computable from logs

---

### 4.5 — Korean pronunciation edge-case fixes
- [ ] Review benchmark results for Korean-specific failures
- [ ] Update custom lexicon with discovered edge cases
- [ ] Add new regression sentences for discovered edge cases
- [ ] Re-run regression suite to confirm no regressions

**Pass criteria:**
- All newly discovered edge cases added to regression set
- Korean G2P regression set: 100% pass rate
- No regressions from previous passing sentences
- Custom lexicon documented

---

### 4.6 — Comprehensive E2E smoke test
- [ ] Expand `scripts/e2e_smoke.sh` to test with diverse inputs:
  - Clean studio recording
  - Phone recording with ambient noise
  - Video with background music
  - Multi-speaker interview/podcast
  - Short reference (5-15s)
  - Long reference (30-120s)
- [ ] Test with various Korean text inputs:
  - Formal speech (합쇼체)
  - Casual speech (해체)
  - Technical text with English terms
  - Numbers, dates, currencies

**Pass criteria:**
- E2E success rate >= 95% across all test inputs
- Each input type tested with at least 2 samples
- Failed cases have clear error classification and actionable messages
- No unhandled crashes

---

### 4.7 — 24-hour soak test
- [ ] Design soak test: continuous random synthesis requests over 24 hours
- [ ] Vary inputs: different speaker profiles, text lengths, text content
- [ ] Monitor for: memory leaks, VRAM leaks, disk space growth, error rate changes

**Pass criteria:**
- 24 hours of continuous operation with no unhandled crashes
- Memory usage remains stable (no monotonic growth)
- Error rate does not increase over time
- All errors are classified and logged

---

## Release Gate Checklist

### Gate A: Pipeline Reliability
- [ ] End-to-end success rate >= 95% on internal media pack
- [ ] No silent failures; every failure mapped to a known error class
- [ ] All API endpoints return correct status codes and error messages

### Gate B: Speech Quality
- [ ] Korean CER < 10% on clean references (via faster-whisper back-transcription)
- [ ] Speaker similarity (ECAPA-TDNN cosine) > 0.75 on clean references
- [ ] DNSMOS > 3.5 on output audio
- [ ] Korean G2P regression set: 100% pass rate (all 50+ sentences)
- [ ] F0 contour correlation > 0.6

---

## Summary: Task Execution Order

| Order | Task | Depends On |
|-------|------|------------|
| 1 | 0.0 — Set up Python 3.13 virtualenv with uv | — |
| 2 | 0.1 — Initialize repo + project config | 0.0 |
| 3 | 0.2 — Create directory structure | 0.1 |
| 4 | 0.3 — Configuration management | 0.2 |
| 5 | 1.1 — Docker Compose stack | 0.3 |
| 6 | 1.2 — Database models | 0.3 |
| 7 | 1.3 — FastAPI app + upload endpoint | 1.1, 1.2 |
| 8 | 1.4 — Celery worker setup | 1.1, 1.2 |
| 9 | 1.5 — FFmpeg extraction + normalization | 0.3 |
| 10 | 1.6 — Silero VAD segmentation | 1.5 |
| 11 | 1.7 — CI pipeline | 0.2 |
| 12 | 2.1 — Background noise detection | 1.6 |
| 13 | 2.2 — Demucs source separation | 2.1 |
| 14 | 2.3 — DeepFilterNet enhancement | 2.2 |
| 15 | 2.4 — Speaker diarization | 1.6 |
| 16 | 2.5 — Segment merging + quality scoring | 2.4 |
| 17 | 2.7 — EBU R128 loudness normalization | 1.5 |
| 18 | 2.6 — Speaker profile builder | 2.3, 2.4, 2.5, 2.7 |
| 19 | 2.8 — Error handling for analysis | 2.6 |
| 20 | 3.1 — TTSBackend + Fish Speech | 0.3 |
| 21 | 3.2 — Korean text normalization | 0.3 |
| 22 | 3.3 — G2P + morphological analysis | 3.2 |
| 23 | 3.4 — Korean regression test set | 3.3 |
| 24 | 3.5 — Synthesis orchestration | 2.6, 3.1, 3.3, 3.6 |
| 25 | 3.6 — Post-processing pipeline | 2.7 |
| 26 | 3.7 — Objective metrics pipeline | 3.5 |
| 27 | 3.8 — End-to-end integration test | 3.5 |
| 28 | 4.1 — CosyVoice integration | 3.1 |
| 29 | 4.2 — Model bake-off benchmark | 4.1, 3.7 |
| 30 | 4.3 — Quality gate | 3.7 |
| 31 | 4.4 — Pipeline monitoring | 3.5 |
| 32 | 4.5 — Korean pronunciation fixes | 4.2, 3.4 |
| 33 | 4.6 — Comprehensive E2E smoke test | 4.3, 4.4 |
| 34 | 4.7 — 24-hour soak test | 4.6 |
| 35 | Release Gate A — Pipeline Reliability | 4.6, 4.7 |
| 36 | Release Gate B — Speech Quality | 4.5, 4.2 |
