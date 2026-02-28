# PLAN_FINAL: Consolidated Execution Plan (Korean Any-Input Voice Cloning TTS)

---

## 1. Product Target

Build a Korean-first voice cloning TTS service that:

1. Accepts any video or audio file as voice reference input.
2. Extracts and cleans the target speaker's voice from noisy/multi-speaker media.
3. Generates new Korean speech in that cloned voice from arbitrary text.

### V1 Constraints

- Zero-shot inference only — no model pretraining from scratch.
- CLI + API interface — no web frontend in V1.
- Single-machine deployment (dev laptop or single GPU server).
- Internal/personal use beta first.

### Non-Goals (V1)

- Singing voice conversion.
- Real-time streaming synthesis.
- Multi-language output (Korean output only; English text handled via transliteration).
- Fine-tuning on user audio (deferred to V2).

---

## 2. Model Selection (Locked for V1)

### Primary: Fish Speech (Apache 2.0)

- Strong Korean/CJK support confirmed by community benchmarks.
- Zero-shot cloning from 10-30s reference audio.
- Active development (regular releases through 2024-2025).
- Built-in API server and WebUI for rapid prototyping.
- GFSQ + Dual AR architecture produces natural prosody.
- 4-8 GB VRAM for inference — runs on consumer GPUs and Apple Silicon (MPS).

### Challenger: CosyVoice (Apache 2.0)

- LLM + Flow Matching architecture.
- Very fast inference.
- 3-10s minimum reference audio (lower barrier).
- Strong Korean support via multilingual pretraining.
- Evaluate during Milestone 4 bake-off.

### Design Pattern

Abstract model behind a `TTSBackend` interface so switching is config-only:

```python
class TTSBackend(Protocol):
    def load_speaker_profile(self, reference_audio: Path) -> SpeakerProfile: ...
    def synthesize(self, text: str, speaker: SpeakerProfile, **kwargs) -> AudioSegment: ...
```

---

## 3. Architecture (Locked for V1)

```
Input (video/audio file + Korean text)
  │
  ├─── AUDIO PREPROCESSING PIPELINE ──────────────────────┐
  │                                                        │
  │  1. FFmpeg: extract audio → mono WAV, 24kHz, 16-bit   │
  │  2. Demucs htdemucs: vocal/music separation           │
  │     (conditional — skip if clean speech detected)      │
  │  3. DeepFilterNet: residual noise removal              │
  │  4. Silero VAD: speech segment detection               │
  │     (threshold=0.5, min_speech=250ms)                  │
  │  5. pyannote 3.1: speaker diarization                  │
  │     (conditional — only for multi-speaker audio)       │
  │  6. Segment merging: 3-15s clips at silence gaps       │
  │  7. EBU R128 loudness normalization (-23 LUFS)         │
  │  8. Quality scoring: SNR + clipping + duration check   │
  │                                                        │
  └────────────────────┬───────────────────────────────────┘
                       │
                       ▼
              Speaker Profile Artifact
              (embedding + best segments + metadata)
                       │
                       │    ┌── TEXT PREPROCESSING ──────────┐
                       │    │                                │
                       │    │  1. Unicode NFC normalization   │
                       │    │  2. Number/date/currency        │
                       │    │     verbalization               │
                       │    │  3. Mecab-ko morphological      │
                       │    │     analysis                    │
                       │    │  4. g2pK grapheme-to-phoneme    │
                       │    │  5. Custom lexicon overrides    │
                       │    │                                │
                       │    └──────────┬─────────────────────┘
                       │               │
                       ▼               ▼
              ┌─────────────────────────────────┐
              │     TTS SYNTHESIS (Fish Speech)  │
              │     conditioned on speaker       │
              │     profile + processed text     │
              └──────────┬──────────────────────┘
                         │
                         ▼
              ┌─────────────────────────────────┐
              │     POST-PROCESSING              │
              │  1. Loudness normalization       │
              │  2. Format conversion            │
              └──────────┬──────────────────────┘
                         │
                         ▼
                    Output Audio
```

### Service Architecture

For V1, a monolithic Python application with background workers:

- **API layer**: FastAPI (REST endpoints)
- **Task queue**: Celery + Redis (preprocessing and synthesis jobs)
- **Metadata store**: PostgreSQL (jobs, speaker profiles)
- **File storage**: Local filesystem with S3-compatible layout (MinIO-ready)
- **Containerization**: Docker Compose (API + worker + Redis + Postgres)

### API Contract (V1)

```
POST /v1/voices/clone
  Input: media file (multipart)
  Output: { job_id, status }

GET /v1/jobs/{job_id}
  Output: { status, stage, progress, errors, speaker_profile_id }

POST /v1/tts/synthesize
  Input: { speaker_profile_id, text, language: "ko" }
  Output: { audio_id, download_url }

GET /v1/audio/{audio_id}
  Output: audio file (WAV/MP3)

GET /v1/voices
  Output: list of speaker profiles

DELETE /v1/voices/{id}
  Output: confirmation + triggers data deletion
```

---

## 4. Korean Language Processing

### Key Challenges

1. **Three-way laryngeal contrast**: lax (ㄱ), tense (ㄲ), aspirated (ㅋ)
2. **Complex G2P rules**: nasalization, lateralization, palatalization, aspiration, tensification, liaison
3. **Seoul dialect prosody**: Accentual Phrase tonal patterns (LHLH / THLH)
4. **Mixed Korean-English text**: brand names, technical terms, code-switching
5. **Number systems**: native Korean (하나, 둘) vs Sino-Korean (일, 이) context-dependent
6. **Honorific levels**: affect prosody and articulation

### Text Processing Pipeline

```
Raw Korean Text
  → Unicode NFC normalization
  → Text normalization:
      3개 → 세 개 | 5시 → 다섯 시 | ₩50,000 → 오만 원
      AI → 에이아이 | 2024년 3월 → 이천이십사년 삼월
  → Mecab-ko morphological analysis (word segmentation + POS)
  → g2pK conversion:
      밖→[박] | 음악을→[으마글] | 학문→[항문]
      신라→[실라] | 같이→[가치] | 좋다→[조타]
  → Phoneme sequence for TTS input
```

### G2P Rule Application Order

Coda neutralization → Liaison → Nasalization → Lateralization → Palatalization → Aspiration → Tensification → Morphophonemic rules

### Libraries

| Library | Purpose |
|---------|---------|
| g2pK | Grapheme-to-phoneme conversion |
| python-mecab-ko | Morphological analysis |
| jamo | Hangul jamo decomposition |
| ko_pron | Korean pronunciation + IPA |

### Korean Regression Test Sentences

Maintain a curated set covering:
- All phonological rules (coda neutralization through tensification)
- Number/counter edge cases (native vs Sino-Korean)
- Mixed Korean-English text
- Honorific level variations
- Long compound words
- Abbreviations and acronyms

---

## 5. Repository Structure

```
tts/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + health endpoints
│   ├── config.py               # Settings via pydantic-settings
│   ├── api/
│   │   ├── routes.py           # API route definitions
│   │   └── schemas.py          # Request/response models
│   ├── pipelines/
│   │   ├── ingest.py           # FFmpeg extraction + normalization
│   │   ├── preprocess.py       # Demucs + DeepFilterNet + VAD
│   │   ├── diarize.py          # Speaker diarization
│   │   ├── analyze.py          # Speaker profile builder
│   │   └── korean_text.py      # Korean text normalization + G2P
│   ├── services/
│   │   ├── tts_backend.py      # TTSBackend protocol + implementations
│   │   ├── synthesis.py        # Orchestrates text processing + TTS
│   │   └── postprocess.py      # Loudness norm + format conversion
│   ├── models/
│   │   └── db.py               # SQLAlchemy models
│   └── workers/
│       └── tasks.py            # Celery task definitions
├── tests/
│   ├── test_ingest.py
│   ├── test_preprocess.py
│   ├── test_korean_text.py
│   ├── test_tts_backend.py
│   ├── test_api.py
│   └── fixtures/               # Sample audio files for testing
├── scripts/
│   ├── e2e_smoke.sh            # End-to-end smoke test
│   └── benchmark.py            # Quality metric evaluation
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── pyproject.toml
└── PLAN_FINAL.md                # This file
```

---

## 6. Useful Korean Datasets

| Dataset | Size | Notes |
|---------|------|-------|
| KSS | ~12 hours | Single female speaker, good for prototyping |
| Zeroth-Korean | ~51 hours | Multi-speaker, CC BY 4.0 |
| CSS10 Korean | ~4 hours | Single speaker, CC0 |
| AIHub Korean | 100s-1000s hours | Multi-speaker, requires registration |
| KsponSpeech | ~1000 hours | Spontaneous conversation |

---

## 7. Execution Plan (4 Milestones, 10 Weeks)

### Milestone 1: Working Skeleton (Weeks 1-2)

**Build:**
- Docker Compose stack: FastAPI + Celery worker + Redis + Postgres
- Upload endpoint → job creation → artifact storage
- FFmpeg audio extraction and normalization pipeline
- Basic Silero VAD segmentation

**Acceptance Criteria:**
- `docker compose up` boots full stack with health checks passing.
- Upload any MP4/WAV → normalized mono WAV persisted to storage.
- VAD returns segment timestamps for sample audio files.
- CI runs lint + unit tests and passes.

**Artifacts:**
- `app/main.py`, `app/pipelines/ingest.py`, `app/pipelines/preprocess.py`
- `tests/test_ingest.py`, `tests/test_preprocess.py`
- `docker/docker-compose.yml`

---

### Milestone 2: Reference Analysis Pipeline (Weeks 3-4)

**Build:**
- Demucs source separation (conditional on background noise detection)
- DeepFilterNet speech enhancement
- pyannote speaker diarization for multi-speaker audio
- Segment quality scoring (SNR proxy + clipping detection + duration)
- Speaker profile artifact builder (best segments + embedding + metadata)

**Acceptance Criteria:**
- On internal test set, dominant speaker selected correctly >= 90%.
- Speaker profile deterministic for same input.
- Noisy reference audio (music, background chatter) cleaned to usable quality.
- Failed analyses return typed error codes with actionable messages.

**Artifacts:**
- `app/pipelines/diarize.py`, `app/pipelines/analyze.py`
- `speaker_profile.json` schema defined and validated
- `tests/test_preprocess.py` (expanded)

---

### Milestone 3: End-to-End TTS (Weeks 5-7)

**Build:**
- Fish Speech integration via `TTSBackend` interface
- Korean text normalization (numbers, dates, currency, abbreviations)
- g2pK + Mecab-ko integration for pronunciation
- Synthesis endpoint → audio generation → download
- Post-processing: loudness normalization + format conversion
**Acceptance Criteria:**
- End-to-end succeeds on curated Korean test set >= 95%.
- p95 synthesis latency < 12s for <= 120 characters on target hardware.
- Korean pronunciation regression set: zero regressions vs baseline.
- Objective metrics pipeline outputs:
  - CER via ASR back-transcription (faster-whisper)
  - Speaker similarity via ECAPA-TDNN cosine similarity

**Artifacts:**
- `app/services/tts_backend.py`, `app/services/synthesis.py`
- `app/pipelines/korean_text.py`, `app/services/postprocess.py`
- `scripts/benchmark.py`
- `tests/test_korean_text.py`, `tests/test_tts_backend.py`

---

### Milestone 4: Hardening + Model Bake-Off (Weeks 8-10)

**Build:**
- Benchmark Fish Speech vs CosyVoice on fixed Korean test suite
  - Metrics: CER, speaker similarity, F0 correlation, DNSMOS, latency, VRAM
- Quality gate: auto-reject low-quality outputs with diagnostics
- Pipeline success rate and latency monitoring
- Korean pronunciation edge-case fixes from benchmark findings

**Acceptance Criteria:**
- Production-default model chosen with benchmark report documenting:
  - Side-by-side metric comparison table
  - Latency/memory tradeoffs
  - Qualitative notes on Korean naturalness
- No P0/P1 failures in regression suite.
- End-to-end success rate >= 95% on internal media pack.
- 24-hour soak test with no unhandled crashes.

**Artifacts:**
- Model bake-off report (markdown)
- `scripts/e2e_smoke.sh` (comprehensive)

---

## 8. Evaluation Framework

### Objective Metrics (Automated)

| Metric | Method | Target |
|--------|--------|--------|
| Intelligibility | CER via faster-whisper back-transcription | < 10% on clean refs |
| Speaker similarity | ECAPA-TDNN embedding cosine similarity | > 0.75 on clean refs |
| Prosody match | F0 contour correlation | > 0.6 |
| Audio quality | DNSMOS proxy score | > 3.5 / 5.0 |
| Latency | Time-to-first-audio | < 8s for short text |
| Korean G2P accuracy | Regression sentence set pass rate | 100% |

### Subjective Evaluation (Manual, Pre-Beta)

- MOS (Mean Opinion Score) for naturalness: target >= 3.5
- SMOS (Speaker Mean Opinion Score) for similarity: target >= 3.5
- Korean native listener review for pronunciation/prosody correctness
- AB preference test: our system vs reference recording

### Test Set Design

Test with diverse reference inputs:
- Clean studio recordings
- Phone recordings with ambient noise
- Video with background music
- Multi-speaker interviews/podcasts
- Short references (5-15s) vs long references (30-120s)
- Different speaking styles (formal, casual, news anchor)

---

## 9. Hardware Requirements

### Development (Apple Silicon / Consumer GPU)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Apple M1 Pro 16GB / RTX 3060 6GB | Apple M2 Pro 32GB / RTX 4070 12GB |
| RAM | 16GB | 32GB |
| Storage | 50GB (models + data) | 100GB+ |

### Apple Silicon Notes

- Fish Speech works on MPS (Metal Performance Shaders).
- Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to prevent OOM.
- M1 Pro 16GB: comfortable for inference.
- Unified memory — no separate VRAM limit.

### Model VRAM Budget

| Component | VRAM |
|-----------|------|
| Fish Speech inference | 4-8 GB |
| Silero VAD | < 1 GB (CPU OK) |
| pyannote diarization | 2-4 GB |
| Demucs | 4-6 GB |
| faster-whisper (large-v3) | 4-8 GB |
| DeepFilterNet | < 1 GB |

Pipeline stages run sequentially, not simultaneously — peak VRAM is the
max of any single stage, not the sum.

---

## 10. Risk Register

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 1 | Noisy references → poor clones | High | Multi-stage preprocessing (Demucs + DeepFilterNet + VAD) + quality scoring + reject with user guidance |
| 2 | Multi-speaker contamination | High | Diarization + speaker-purity checks + user speaker selection |
| 3 | Korean pronunciation errors | Medium | g2pK + custom lexicon overrides + regression sentence set |
| 4 | Fish Speech Korean quality insufficient | High | CosyVoice as evaluated challenger; TTSBackend abstraction makes switching config-only |
| 5 | Apple Silicon compatibility issues | Medium | Test on MPS early (Milestone 1); CPU fallback for preprocessing |

---

## 11. Dependencies

```
# Core
torch>=2.0
torchaudio>=2.0
fastapi>=0.100
uvicorn>=0.20
celery>=5.3
redis>=5.0

# Audio Processing
demucs>=4.0
deepfilternet>=0.5
silero-vad>=5.0
pyloudnorm>=0.1
soxr>=0.3
soundfile>=0.12

# Speech Processing
pyannote.audio>=3.1
faster-whisper>=1.0
speechbrain>=1.0           # ECAPA-TDNN embeddings

# Korean NLP
g2pk>=0.9
python-mecab-ko>=1.0
jamo>=0.4

# TTS Engine
# fish-speech (install from source)

# Storage
sqlalchemy>=2.0
psycopg2-binary>=2.9

# Testing
pytest>=7.0
httpx>=0.24                # Async test client for FastAPI
```

---

## 12. Immediate 7-Day Kickoff

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Scaffold repo structure + pyproject.toml + Docker Compose | Bootable stack |
| 2 | Implement upload endpoint + job table + local file storage | Files persist after upload |
| 3 | Add FFmpeg normalization + Silero VAD segmentation | Normalized WAV + segment timestamps |
| 4 | Add Demucs source separation + DeepFilterNet | Clean vocals from noisy test audio |
| 5 | Add pyannote diarization + speaker selection logic | Multi-speaker audio handled |
| 6 | Define speaker_profile.json schema + quality scoring | Profile artifacts generated |
| 7 | Run E2E smoke test on 5 diverse samples + fix top blockers | Smoke test passing |

---

## 13. Release Gates

### Gate A: Pipeline Reliability
- End-to-end success rate >= 95% on internal media pack.
- No silent failures; every failure mapped to a known error class.

### Gate B: Speech Quality
- Korean intelligibility (CER) meets baseline threshold on clean references.
- Speaker similarity proxy exceeds internal threshold.
- Korean G2P regression set: 100% pass rate.

---

## 14. V2 Roadmap 

- Web frontend (React + wavesurfer.js)
- GPT-SoVITS fine-tuning path for higher quality with more reference audio
- Streaming synthesis via WebSocket
- Batch synthesis API
- Multi-language output (English, Japanese)
- Prosody transfer controls (speed, pitch, emotion)
- C2PA provenance metadata
