"""Celery task definitions.

Defines the voice cloning processing task that orchestrates
the full analysis pipeline as a background job.
"""

from celery import Celery
