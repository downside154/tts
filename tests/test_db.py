"""Tests for database models and session management.

Covers engine creation, session factory, get_db generator,
and model instantiation.
"""

import contextlib
from unittest.mock import patch

from app.models.db import (
    get_db,
    get_engine,
    get_session_factory,
)


class TestGetEngine:
    def test_returns_engine(self) -> None:
        """get_engine returns a SQLAlchemy engine."""
        get_engine.cache_clear()
        with patch("app.models.db.settings") as mock_settings:
            mock_settings.database_url = "sqlite:///:memory:"
            engine = get_engine()
            assert engine is not None
        get_engine.cache_clear()


class TestGetSessionFactory:
    def test_returns_sessionmaker(self) -> None:
        """get_session_factory returns a sessionmaker bound to the engine."""
        get_engine.cache_clear()
        with patch("app.models.db.settings") as mock_settings:
            mock_settings.database_url = "sqlite:///:memory:"
            factory = get_session_factory()
            assert factory is not None
            session = factory()
            session.close()
        get_engine.cache_clear()


class TestGetDb:
    def test_yields_and_closes_session(self) -> None:
        """get_db yields a session and closes it after use."""
        get_engine.cache_clear()
        with patch("app.models.db.settings") as mock_settings:
            mock_settings.database_url = "sqlite:///:memory:"
            gen = get_db()
            session = next(gen)
            assert session is not None
            with contextlib.suppress(StopIteration):
                next(gen)
        get_engine.cache_clear()


class TestMainLifespan:
    def test_lifespan_creates_tables(self) -> None:
        """Lifespan event creates database tables on startup."""
        import asyncio
        from unittest.mock import MagicMock

        from sqlalchemy import inspect

        from app.main import lifespan

        get_engine.cache_clear()
        with patch("app.models.db.settings") as mock_settings:
            mock_settings.database_url = "sqlite:///:memory:"

            async def run():
                mock_app = MagicMock()
                async with lifespan(mock_app):
                    engine = get_engine()
                    inspector = inspect(engine)
                    tables = inspector.get_table_names()
                    assert "jobs" in tables

            asyncio.run(run())
        get_engine.cache_clear()
