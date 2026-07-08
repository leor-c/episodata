"""In-memory buffers for ongoing episodes, shared by persistent backends.

Ongoing (not yet finalized) episodes are buffered in memory as per-field
chunk lists. Backends spill them to ``.npz`` files on flush for crash
recovery and write them into their final layout on finalize.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np

from .base import Selection


def check_lengths(fields: Mapping[str, np.ndarray]) -> int:
    """Validate that all appended fields share one length; return it."""
    lengths = {k: len(v) for k, v in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"appended fields must share one length, got {lengths}")
    return next(iter(lengths.values()))


def atomic_write(path: str, write: Callable) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_npz(path: str, arrays: Mapping[str, np.ndarray]) -> None:
    atomic_write(path, lambda f: np.savez_compressed(f, **arrays))


class EpisodeBuffers:
    """Per-episode field buffers: chunk lists with lazy consolidation."""

    def __init__(self) -> None:
        self._chunks: dict[int, dict[str, list[np.ndarray]]] = {}

    def __contains__(self, episode_id: int) -> bool:
        return episode_id in self._chunks

    def __iter__(self) -> Iterator[int]:
        return iter(self._chunks)

    def create(self, episode_id: int) -> None:
        self._chunks[episode_id] = {}

    def append(self, episode_id: int, fields: Mapping[str, np.ndarray]) -> int:
        """Append one multi-field chunk; returns the number of steps added."""
        n = check_lengths(fields)
        buffer = self._chunks[episode_id]
        for key, arr in fields.items():
            buffer.setdefault(key, []).append(np.asarray(arr))
        return n

    def read(self, field_ids: Sequence[str], selection: Selection) -> dict[str, np.ndarray]:
        buffer = self._chunks[selection.episode_id]
        out: dict[str, np.ndarray] = {}
        for key in field_ids:
            chunks = buffer.get(key)
            if not chunks:
                raise KeyError(f"episode {selection.episode_id} has no field {key!r}")
            if len(chunks) > 1:
                buffer[key] = chunks = [np.concatenate(chunks, axis=0)]
            out[key] = chunks[0][selection.start : selection.stop]
        return out

    def arrays(self, episode_id: int) -> dict[str, np.ndarray]:
        """The buffered fields, consolidated to one array each."""
        buffer = self._chunks[episode_id]
        for key, chunks in buffer.items():
            if len(chunks) > 1:
                buffer[key] = [np.concatenate(chunks, axis=0)]
        return {key: chunks[0] for key, chunks in buffer.items()}

    def is_empty(self, episode_id: int) -> bool:
        return not self._chunks[episode_id]

    def drop(self, episode_id: int) -> None:
        del self._chunks[episode_id]

    def save(self, episode_id: int, path: str) -> None:
        save_npz(path, self.arrays(episode_id))

    def restore(self, episode_id: int, path: str) -> int:
        """Restore a spilled buffer from ``path``; returns its length."""
        with np.load(path) as archive:
            self._chunks[episode_id] = {k: [archive[k]] for k in archive.files}
        buffer = self._chunks[episode_id]
        return len(next(iter(buffer.values()))[0]) if buffer else 0
