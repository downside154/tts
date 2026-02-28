"""SQLAlchemy 2.0 database models.

Defines the Job, SpeakerProfile, and AudioOutput tables
with their relationships and constraints.
"""

import enum
from collections.abc import Iterator
from datetime import datetime
from functools import lru_cache

from sqlalchemy import (
    DateTime,
    Engine,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import settings


@lru_cache
def get_engine() -> Engine:
    return create_engine(settings.database_url)


def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine())


def get_db() -> Iterator[Session]:
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


class Base(DeclarativeBase):
    pass


class JobStatus(enum.StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SpeakerProfile(Base):
    __tablename__ = "speaker_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_path: Mapped[str | None] = mapped_column(String(512))
    segments_json: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    jobs: Mapped[list["Job"]] = relationship(back_populates="speaker_profile")
    audio_outputs: Mapped[list["AudioOutput"]] = relationship(
        back_populates="speaker_profile"
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.PENDING, nullable=False
    )
    stage: Mapped[str | None] = mapped_column(String(100))
    progress: Mapped[float | None] = mapped_column(Float)
    error_code: Mapped[str | None] = mapped_column(String(50))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    input_file_path: Mapped[str | None] = mapped_column(String(512))
    speaker_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("speaker_profiles.id")
    )

    speaker_profile: Mapped[SpeakerProfile | None] = relationship(
        back_populates="jobs"
    )


class AudioOutput(Base):
    __tablename__ = "audio_outputs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"))
    speaker_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("speaker_profiles.id")
    )
    text_input: Mapped[str | None] = mapped_column(Text)
    file_path: Mapped[str | None] = mapped_column(String(512))
    format: Mapped[str | None] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    job: Mapped[Job | None] = relationship()
    speaker_profile: Mapped[SpeakerProfile | None] = relationship(
        back_populates="audio_outputs"
    )
