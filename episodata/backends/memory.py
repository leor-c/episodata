"""In-memory storage backend: episodes held as per-field numpy arrays.

Good for toy datasets, tests, and online replay-buffer style usage. Not
persistent.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..schema import DatasetSchema
from .base import Payload, Selection, StorageBackend, register_backend


@dataclasses.dataclass
class _Episode:
    chunks: dict[str, list[np.ndarray]]
    length: int = 0
    terminated: bool = False
    truncated: bool = False
    ongoing: bool = True


@register_backend
class MemoryBackend(StorageBackend):
    name = "memory"

    def __init__(self, schema: DatasetSchema):
        self._schema = schema
        self._episodes: list[_Episode] = []

    @classmethod
    def create(cls, schema: DatasetSchema, path: str | None = None, **options: Any) -> "MemoryBackend":
        if path is not None:
            raise ValueError("MemoryBackend is not persistent; do not pass a path")
        return cls(schema)

    @classmethod
    def open(cls, path: str) -> "MemoryBackend":
        raise NotImplementedError("MemoryBackend cannot be reopened from a path")

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    def write_schema(self, schema: DatasetSchema) -> None:
        self._schema = schema

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def episode_length(self, episode_id: int) -> int:
        return self._episodes[episode_id].length

    def episode_terminated(self, episode_id: int) -> bool:
        return self._episodes[episode_id].terminated

    def episode_truncated(self, episode_id: int) -> bool:
        return self._episodes[episode_id].truncated

    def episode_ongoing(self, episode_id: int) -> bool:
        return self._episodes[episode_id].ongoing

    def read_fields(self, field_ids: Sequence[str], selection: Selection) -> Payload:
        episode = self._episodes[selection.episode_id]
        self._consolidate(episode)
        out: dict[str, np.ndarray] = {}
        for key in field_ids:
            chunks = episode.chunks.get(key)
            if not chunks:
                raise KeyError(f"episode {selection.episode_id} has no field {key!r}")
            out[key] = chunks[0][selection.start : selection.stop]
        return out

    def create_episode(self) -> int:
        self._episodes.append(_Episode(chunks={k: [] for k in self._schema.fields}))
        self._touch()
        return len(self._episodes) - 1

    def append_steps(self, episode_id: int, fields: Mapping[str, np.ndarray]) -> None:
        episode = self._episodes[episode_id]
        if not episode.ongoing:
            raise ValueError(f"episode {episode_id} is finalized")
        n = _check_lengths(fields)
        for key, arr in fields.items():
            episode.chunks.setdefault(key, []).append(np.asarray(arr))
        episode.length += n
        self._touch()

    def finalize_episode(self, episode_id: int, terminated: bool, truncated: bool) -> None:
        episode = self._episodes[episode_id]
        episode.ongoing = False
        episode.terminated = terminated
        episode.truncated = truncated
        self._consolidate(episode)
        self._touch()

    @staticmethod
    def _consolidate(episode: _Episode) -> None:
        for key, chunks in episode.chunks.items():
            if len(chunks) > 1:
                episode.chunks[key] = [np.concatenate(chunks, axis=0)]


def _check_lengths(fields: Mapping[str, np.ndarray]) -> int:
    lengths = {k: len(v) for k, v in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"appended fields must share one length, got {lengths}")
    return next(iter(lengths.values()))