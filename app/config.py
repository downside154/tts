"""Application configuration management.

Uses pydantic-settings to load configuration from environment variables
and .env files with typed validation.
"""

from pydantic_settings import BaseSettings
