"""Storage backends.

Importing this package registers the built-in backends. Third-party
backends register themselves with :func:`register_backend`.
"""

from .base import (
    Payload,
    Selection,
    SpaceBlock,
    StorageBackend,
    get_backend,
    normalize_payload,
    register_backend,
    register_missing_backend,
)
from .memory import MemoryBackend
from .npz import NpzDirectoryBackend

try:
    from .zarr import ZarrBackend
except ImportError:
    ZarrBackend = None  # type: ignore[assignment, misc]
    register_missing_backend(
        "zarr",
        "it requires the optional 'zarr' package (Python >= 3.11); "
        'install with pip install "episodata[zarr]"',
    )

__all__ = [
    "MemoryBackend",
    "NpzDirectoryBackend",
    "ZarrBackend",
    "Payload",
    "Selection",
    "SpaceBlock",
    "StorageBackend",
    "get_backend",
    "normalize_payload",
    "register_backend",
]