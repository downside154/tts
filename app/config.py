"""Application configuration management.

Uses pydantic-settings to load configuration from environment variables
and .env files with typed validation.
"""

from enum import StrEnum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DeviceType(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Database
    database_url: str = "postgresql://tts:tts@localhost:5432/tts"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Storage
    storage_path: Path = Path("./storage")

    # Model paths
    fish_speech_model_path: Path = Path("./models/fish-speech")
    cosyvoice_model_path: Path = Path("./models/cosyvoice")

    # FFmpeg
    ffmpeg_path: str = "ffmpeg"

    # Device
    device: DeviceType = DeviceType.MPS

    # Debug
    debug: bool = True


settings = Settings()
