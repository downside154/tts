"""Audio ingestion pipeline.

Handles extraction of audio from video/audio files using FFmpeg,
converting to a standardized format (mono WAV, 24kHz, 16-bit PCM).
"""

from pathlib import Path
