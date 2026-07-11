"""Chunked single-store backend built on Zarr v3: all episodes in one store.

Layout::

    <root>/
        manifest.json            # backend name + logical schema (constant size)
        data.zarr/
            fields/<field_key>   # [T_total, *field.shape], chunked on time
            index/start          # [N] int64, global start row (-1 while ongoing)
            index/length         # [N] int64
            index/terminated     # [N] bool
            index/truncated      # [N] bool
            index/ongoing        # [N] bool
            index/present        # [N, F] bool, schema field order
        pending/                 # ongoing-episode buffers spilled on flush
            ep_000042.npz

Each field is one array concatenated along time across episodes; ongoing
episodes are buffered in memory and appended contiguously on finalize,
when their ``start`` is assigned — so interleaved episodes may finalize in
any order. Finalize and open cost O(1) in the number of episodes: index
updates are point writes into chunked arrays, never a manifest rewrite,
and temporal slicing decodes only the chunks overlapping the selection.
This is the backend for datasets that outgrow one-file-per-episode
(``npz_directory``); migrate with ``Dataset.copy_to``.

Fields an episode never received stay untouched in its row range (zarr
chunks are materialized only when written); ``index/present`` records which
fields each episode actually has so reads of absent fields raise KeyError.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import zarr

from ..schema import DatasetSchema
from ._buffers import EpisodeBuffers, atomic_write
from .base import Selection, StorageBackend, register_backend

_FORMAT_VERSION = 1
_INDEX_CHUNK = 65536
_DEFAULT_CHUNK_BYTES = 4 << 20
_MAX_CHUNK_STEPS = 65536


@register_backend
class ZarrBackend(StorageBackend):
    name = "zarr"

    def __init__(self, root: str, schema: DatasetSchema, group: zarr.Group):
        self._root = root
        self._schema = schema
        self._group = group
        self._field_keys: list[str] = list(group.attrs["field_keys"])
        self._buffers = EpisodeBuffers()
        # In-memory mirror of index/; disk rows are written on finalize/flush.
        self._start: list[int] = []
        self._length: list[int] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._ongoing: list[bool] = []
        self._present: list[set[str]] = []
        self._total = group[f"fields/{self._field_keys[0]}"].shape[0] if self._field_keys else 0

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        schema: DatasetSchema,
        path: str | None = None,
        *,
        chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
        shard_bytes: int | None = None,
        **options: Any,
    ) -> "ZarrBackend":
        """Create a new store.

        ``chunk_bytes`` targets the compressed-unit size along time per
        field (the read granularity); ``shard_bytes`` optionally packs many
        chunks into one file (Zarr v3 sharding) to cap file count.
        """
        if path is None:
            raise ValueError("ZarrBackend requires a path")
        if os.path.exists(os.path.join(path, "manifest.json")):
            raise FileExistsError(f"a dataset already exists at {path}")
        os.makedirs(os.path.join(path, "pending"), exist_ok=True)
        group = zarr.open_group(os.path.join(path, "data.zarr"), mode="w")
        field_keys = schema.field_keys()
        group.attrs["field_keys"] = field_keys
        for key in field_keys:
            spec = schema.field(key)
            shape = tuple(spec.shape)
            step_bytes = max(1, np.dtype(spec.dtype).itemsize * math.prod(shape))
            chunk_t = max(1, min(chunk_bytes // step_bytes, _MAX_CHUNK_STEPS))
            shards = None
            if shard_bytes is not None:
                shard_t = chunk_t * max(1, shard_bytes // (chunk_t * step_bytes))
                shards = (shard_t, *shape)
            group.create_array(
                f"fields/{key}",
                shape=(0, *shape),
                chunks=(chunk_t, *shape),
                shards=shards,
                dtype=spec.dtype,
            )
        # Fill values encode a freshly created, still-empty ongoing episode,
        # so an index row is valid even if a crash prevents its point write.
        for name, dtype, fill in (
            ("start", "int64", -1),
            ("length", "int64", 0),
            ("terminated", "bool", False),
            ("truncated", "bool", False),
            ("ongoing", "bool", True),
        ):
            group.create_array(
                f"index/{name}", shape=(0,), chunks=(_INDEX_CHUNK,), dtype=dtype, fill_value=fill
            )
        group.create_array(
            "index/present",
            shape=(0, len(field_keys)),
            chunks=(_INDEX_CHUNK, max(1, len(field_keys))),
            dtype="bool",
            fill_value=False,
        )
        backend = cls(path, schema, group)
        backend._write_manifest()
        return backend

    @classmethod
    def open(cls, path: str) -> "ZarrBackend":
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        if manifest["backend"] != cls.name:
            raise ValueError(
                f"dataset at {path} uses backend {manifest['backend']!r}, not {cls.name!r}"
            )
        schema = DatasetSchema.from_dict(manifest["schema"])
        group = zarr.open_group(os.path.join(path, "data.zarr"), mode="a")
        backend = cls(path, schema, group)
        index = group["index"]
        backend._start = index["start"][:].tolist()
        backend._length = index["length"][:].tolist()
        backend._terminated = index["terminated"][:].tolist()
        backend._truncated = index["truncated"][:].tolist()
        backend._ongoing = index["ongoing"][:].tolist()
        present = index["present"][:]
        backend._present = [
            {k for k, p in zip(backend._field_keys, row) if p} for row in present
        ]
        # Ongoing episodes are buffered in memory; restore what flush() spilled
        # (or reset the row if the process died before any flush).
        for episode_id, ongoing in enumerate(backend._ongoing):
            if not ongoing:
                continue
            pending = backend._pending_path(episode_id)
            if os.path.exists(pending):
                backend._length[episode_id] = backend._buffers.restore(episode_id, pending)
                backend._present[episode_id] = set(backend._buffers.arrays(episode_id))
            else:
                backend._length[episode_id] = 0
                backend._present[episode_id] = set()
                backend._buffers.create(episode_id)
        return backend

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    # -- episode index -------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        return len(self._start)

    def episode_length(self, episode_id: int) -> int:
        return self._length[episode_id]

    def episode_terminated(self, episode_id: int) -> bool:
        return self._terminated[episode_id]

    def episode_truncated(self, episode_id: int) -> bool:
        return self._truncated[episode_id]

    def episode_ongoing(self, episode_id: int) -> bool:
        return self._ongoing[episode_id]

    # -- reads ---------------------------------------------------------------

    def read_fields(
        self, field_ids: Sequence[str], selection: Selection
    ) -> Mapping[str, np.ndarray]:
        episode_id = selection.episode_id
        if episode_id in self._buffers:
            return self._buffers.read(field_ids, selection)
        start = self._start[episode_id]
        out: dict[str, np.ndarray] = {}
        for key in field_ids:
            if key not in self._present[episode_id]:
                raise KeyError(f"episode {episode_id} has no field {key!r}")
            array = self._group[f"fields/{key}"]
            out[key] = array[start + selection.start : start + selection.stop]
        return out

    # -- writes ----------------------------------------------------------------

    def create_episode(self) -> int:
        episode_id = len(self._start)
        self._start.append(-1)
        self._length.append(0)
        self._terminated.append(False)
        self._truncated.append(False)
        self._ongoing.append(True)
        self._present.append(set())
        self._buffers.create(episode_id)
        self._touch()
        return episode_id

    def append_steps(self, episode_id: int, fields: Mapping[str, np.ndarray]) -> None:
        if not self._ongoing[episode_id]:
            raise ValueError(f"episode {episode_id} is finalized")
        self._length[episode_id] += self._buffers.append(episode_id, fields)
        self._present[episode_id].update(fields)
        self._touch()

    def finalize_episode(self, episode_id: int, terminated: bool, truncated: bool) -> None:
        if not self._ongoing[episode_id]:
            raise ValueError(f"episode {episode_id} is already finalized")
        arrays = self._buffers.arrays(episode_id)
        length = self._length[episode_id]
        ragged = {k: len(v) for k, v in arrays.items() if len(v) != length}
        if ragged:
            raise ValueError(
                f"episode {episode_id} has fields shorter than its length "
                f"{length}: {ragged} — all present fields must cover every step"
            )
        start = self._total
        stop = start + length
        for key in self._field_keys:
            array = self._group[f"fields/{key}"]
            array.resize((stop, *array.shape[1:]))
            if key in arrays and start < stop:
                array[start:stop] = arrays[key]
        self._total = stop
        self._start[episode_id] = start
        self._terminated[episode_id] = terminated
        self._truncated[episode_id] = truncated
        self._ongoing[episode_id] = False
        self._sync_index_len()
        self._write_index_row(episode_id)
        self._buffers.drop(episode_id)
        pending = self._pending_path(episode_id)
        if os.path.exists(pending):
            os.remove(pending)
        self._touch()

    # -- persistence -------------------------------------------------------------

    def flush(self) -> None:
        self._sync_index_len()
        for episode_id in self._buffers:
            self._buffers.save(episode_id, self._pending_path(episode_id))
            self._write_index_row(episode_id)

    def _sync_index_len(self) -> None:
        """Grow the on-disk index to the in-memory episode count. New rows
        need no write: their fill values already read as an empty ongoing
        episode."""
        n = len(self._start)
        index = self._group["index"]
        if index["start"].shape[0] >= n:
            return
        for name in ("start", "length", "terminated", "truncated", "ongoing"):
            index[name].resize((n,))
        index["present"].resize((n, index["present"].shape[1]))

    def _write_index_row(self, episode_id: int) -> None:
        index = self._group["index"]
        index["start"][episode_id] = self._start[episode_id]
        index["length"][episode_id] = self._length[episode_id]
        index["terminated"][episode_id] = self._terminated[episode_id]
        index["truncated"][episode_id] = self._truncated[episode_id]
        index["ongoing"][episode_id] = self._ongoing[episode_id]
        present = self._present[episode_id]
        index["present"][episode_id] = np.array(
            [key in present for key in self._field_keys], dtype=bool
        )

    def _pending_path(self, episode_id: int) -> str:
        return os.path.join(self._root, "pending", f"ep_{episode_id:06d}.npz")

    def _write_manifest(self) -> None:
        manifest = {
            "format_version": _FORMAT_VERSION,
            "backend": self.name,
            "schema": self._schema.to_dict(),
        }
        path = os.path.join(self._root, "manifest.json")
        atomic_write(path, lambda f: f.write(json.dumps(manifest, indent=2).encode()))
