"""Audio preprocessing pipeline.

Includes VAD segmentation (Silero), background noise detection,
source separation (Demucs), speech enhancement (DeepFilterNet),
and loudness normalization (EBU R128).
"""

from pathlib import Path
