"""Storage backends.

Importing this package registers the built-in backends. Third-party
backends register themselves with :func:`register_backend`.
"""

from .base import (
    Selection,
    StorageBackend,
    get_backend,
    register_backend,
    register_missing_backend,
)
from .memory import MemoryBackend
from .npz import NpzDirectoryBackend
from .zarr import ZarrBackend

__all__ = [
    "MemoryBackend",
    "NpzDirectoryBackend",
    "ZarrBackend",
    "Selection",
    "StorageBackend",
    "get_backend",
    "register_backend",
]