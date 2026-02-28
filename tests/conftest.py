"""Shared test configuration.

Injects mock modules for optional ML dependencies (demucs, deepfilternet,
speechbrain, pyannote) so that ``unittest.mock.patch()`` targets resolve
even when the real packages are not installed (e.g. CI without the ``ml``
dependency group).
"""

import sys
from unittest.mock import MagicMock

_OPTIONAL_ML_PACKAGES: dict[str, list[str]] = {
    "demucs": [
        "demucs",
        "demucs.pretrained",
        "demucs.apply",
    ],
    "df": [
        "df",
        "df.enhance",
    ],
    "speechbrain": [
        "speechbrain",
        "speechbrain.inference",
        "speechbrain.inference.speaker",
    ],
    "pyannote": [
        "pyannote",
        "pyannote.audio",
    ],
}


def _inject_mock_modules() -> None:
    """Add MagicMock modules to sys.modules for packages that aren't installed.

    Uses ``importlib.util.find_spec`` to avoid triggering side effects from
    actually importing the package (some ML packages run heavy init code).
    """
    import importlib.util

    for base_package, submodules in _OPTIONAL_ML_PACKAGES.items():
        if importlib.util.find_spec(base_package) is None:
            for mod in submodules:
                if mod not in sys.modules:
                    sys.modules[mod] = MagicMock()


_inject_mock_modules()
