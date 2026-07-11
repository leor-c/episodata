"""Persistent directory backend: one ``.npz`` file per episode.

Layout::

    <root>/
        manifest.json           # logical schema + storage manifest
        episodes/
            ep_000000.npz       # one compressed array per field
            ep_000001.npz
            ...

Field selection is pushed down naturally: npz members are individual zip
entries, so only the requested fields are decompressed. Temporal slicing
currently decodes the whole per-episode field before slicing — finer
chunking is a backend-internal concern that can change without touching
the logical API.

Ongoing (not yet finalized) episodes are buffered in memory and written to
disk on finalize or flush.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..schema import DatasetSchema
from ._buffers import EpisodeBuffers, atomic_write
from .base import Selection, StorageBackend, register_backend

_FORMAT_VERSION = 1


@dataclasses.dataclass
class _EpisodeRecord:
    file: str
    length: int
    terminated: bool = False
    truncated: bool = False
    ongoing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "_EpisodeRecord":
        return cls(**d)


@register_backend
class NpzDirectoryBackend(StorageBackend):
    name = "npz_directory"

    def __init__(self, root: str, schema: DatasetSchema, records: list[_EpisodeRecord]):
        self._root = root
        self._schema = schema
        self._records = records
        self._buffers = EpisodeBuffers()

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def create(cls, schema: DatasetSchema, path: str | None = None, **options: Any) -> "NpzDirectoryBackend":
        if path is None:
            raise ValueError("NpzDirectoryBackend requires a path")
        os.makedirs(os.path.join(path, "episodes"), exist_ok=True)
        if os.path.exists(os.path.join(path, "manifest.json")):
            raise FileExistsError(f"a dataset already exists at {path}")
        backend = cls(path, schema, [])
        backend.flush()
        return backend

    @classmethod
    def open(cls, path: str) -> "NpzDirectoryBackend":
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        if manifest["backend"] != cls.name:
            raise ValueError(
                f"dataset at {path} uses backend {manifest['backend']!r}, not {cls.name!r}"
            )
        schema = DatasetSchema.from_dict(manifest["schema"])
        records = [_EpisodeRecord.from_dict(r) for r in manifest["storage"]["episodes"]]
        backend = cls(path, schema, records)
        # Ongoing episodes are buffered in memory; restore what flush() saved
        # (or reset the record if the process died before any flush).
        for episode_id, record in enumerate(records):
            if not record.ongoing:
                continue
            file_path = os.path.join(path, record.file)
            if os.path.exists(file_path):
                backend._buffers.restore(episode_id, file_path)
            else:
                record.length = 0
                backend._buffers.create(episode_id)
        return backend

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    def write_schema(self, schema: DatasetSchema) -> None:
        self._schema = schema
        self.flush()

    # -- episode index -------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        return len(self._records)

    def episode_length(self, episode_id: int) -> int:
        return self._records[episode_id].length

    def episode_terminated(self, episode_id: int) -> bool:
        return self._records[episode_id].terminated

    def episode_truncated(self, episode_id: int) -> bool:
        return self._records[episode_id].truncated

    def episode_ongoing(self, episode_id: int) -> bool:
        return self._records[episode_id].ongoing

    # -- reads ---------------------------------------------------------------

    def read_fields(
        self, field_ids: Sequence[str], selection: Selection
    ) -> Mapping[str, np.ndarray]:
        episode_id = selection.episode_id
        if episode_id in self._buffers:
            return self._buffers.read(field_ids, selection)
        path = os.path.join(self._root, self._records[episode_id].file)
        out: dict[str, np.ndarray] = {}
        with np.load(path) as archive:
            for key in field_ids:
                if key not in archive:
                    raise KeyError(f"episode {episode_id} has no field {key!r}")
                out[key] = archive[key][selection.start : selection.stop]
        return out

    # -- writes ----------------------------------------------------------------

    def create_episode(self) -> int:
        episode_id = len(self._records)
        self._records.append(
            _EpisodeRecord(file=os.path.join("episodes", f"ep_{episode_id:06d}.npz"), length=0)
        )
        self._buffers.create(episode_id)
        self._touch()
        return episode_id

    def append_steps(self, episode_id: int, fields: Mapping[str, np.ndarray]) -> None:
        record = self._records[episode_id]
        if not record.ongoing:
            raise ValueError(f"episode {episode_id} is finalized")
        record.length += self._buffers.append(episode_id, fields)
        self._touch()

    def finalize_episode(self, episode_id: int, terminated: bool, truncated: bool) -> None:
        record = self._records[episode_id]
        record.terminated = terminated
        record.truncated = truncated
        record.ongoing = False
        self._write_episode(episode_id)
        self._buffers.drop(episode_id)
        self._touch()
        self.flush()

    def _write_episode(self, episode_id: int) -> None:
        path = os.path.join(self._root, self._records[episode_id].file)
        self._buffers.save(episode_id, path)

    # -- persistence -------------------------------------------------------------

    def flush(self) -> None:
        for episode_id in self._buffers:
            if not self._buffers.is_empty(episode_id):
                self._write_episode(episode_id)
        manifest = {
            "format_version": _FORMAT_VERSION,
            "backend": self.name,
            "schema": self._schema.to_dict(),
            "storage": {"episodes": [r.to_dict() for r in self._records]},
        }
        path = os.path.join(self._root, "manifest.json")
        atomic_write(path, lambda f: f.write(json.dumps(manifest, indent=2).encode()))