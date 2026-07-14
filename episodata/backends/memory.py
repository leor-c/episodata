"""In-memory storage backend: episodes held as per-field numpy arrays.

Good for toy datasets, tests, and online replay-buffer style usage. Not
persistent.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import DatasetSchema
from ._buffers import EpisodeBuffers
from .base import Selection, StorageBackend, register_backend


@dataclasses.dataclass
class _Episode:
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
        # Field data lives in the shared chunk buffers for the episode's
        # whole life — memory *is* the final layout, so nothing moves on
        # finalize (beyond consolidation).
        self._buffers = EpisodeBuffers()

    @classmethod
    def create(
        cls, schema: DatasetSchema, path: str | Path | None = None, **options: Any
    ) -> "MemoryBackend":
        if path is not None:
            raise ValueError("MemoryBackend is not persistent; do not pass a path")
        return cls(schema)

    @classmethod
    def open(cls, path: str | Path) -> "MemoryBackend":
        raise NotImplementedError("MemoryBackend cannot be reopened from a path")

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

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

    def read_fields(
        self, field_ids: Sequence[str], selection: Selection
    ) -> Mapping[str, np.ndarray]:
        return self._buffers.read(field_ids, selection)

    def create_episode(self) -> int:
        self._episodes.append(_Episode())
        self._buffers.create(len(self._episodes) - 1)
        self._touch()
        return len(self._episodes) - 1

    def append_steps(self, episode_id: int, fields: Mapping[str, np.ndarray]) -> None:
        episode = self._episodes[episode_id]
        if not episode.ongoing:
            raise ValueError(f"episode {episode_id} is finalized")
        episode.length += self._buffers.append(episode_id, fields)
        self._touch()

    def finalize_episode(self, episode_id: int, terminated: bool, truncated: bool) -> None:
        episode = self._episodes[episode_id]
        episode.ongoing = False
        episode.terminated = terminated
        episode.truncated = truncated
        self._buffers.arrays(episode_id)  # consolidate chunks for reads
        self._touch()
