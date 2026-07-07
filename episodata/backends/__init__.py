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
)
from .memory import MemoryBackend
from .npz import NpzDirectoryBackend

__all__ = [
    "MemoryBackend",
    "NpzDirectoryBackend",
    "Payload",
    "Selection",
    "SpaceBlock",
    "StorageBackend",
    "get_backend",
    "normalize_payload",
    "register_backend",
]