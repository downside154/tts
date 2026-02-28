"""Speech synthesis orchestration service.

Coordinates the full synthesis pipeline: loading speaker profiles,
normalizing text, running G2P conversion, invoking the TTS backend,
and post-processing the output audio.
"""

from pathlib import Path
