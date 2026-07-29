"""Chunked single-store backend built on Zarr v3 (via TensorStore): all
episodes in one store.

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

I/O goes through TensorStore's zarr3 driver rather than zarr-python: reads
here are near-always batched (many selections gathered in one call — see
:meth:`read_fields`), and zarr-python's synchronous API pays a large fixed
dispatch cost per call regardless of batch size, which TensorStore avoids.
The two are format-compatible (TensorStore reads/writes the same on-disk
zarr v3 layout zarr-python does) — this is purely an I/O-engine choice, not
a storage format change. Each field/index array is opened independently by
its own kvstore path; ``fields/`` and ``index/`` are plain directories, not
zarr *groups* — nothing here ever addresses ``data.zarr`` as a whole group,
so no root group metadata is written.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tensorstore as ts

from ..schema import DatasetSchema
from ._buffers import EpisodeBuffers, atomic_write
from .base import Selection, StorageBackend, register_backend

_FORMAT_VERSION = 1
_INDEX_CHUNK = 65536
_DEFAULT_CHUNK_BYTES = 4 << 20
_MAX_CHUNK_STEPS = 65536

_INDEX_FIELDS: tuple[tuple[str, str, Any], ...] = (
    # Fill values encode a freshly created, still-empty ongoing episode, so
    # an index row is valid even if a crash prevents its point write.
    ("start", "int64", -1),
    ("length", "int64", 0),
    ("terminated", "bool", False),
    ("truncated", "bool", False),
    ("ongoing", "bool", True),
)


def _create_array(
    path: str,
    *,
    shape: Sequence[int],
    chunk_shape: Sequence[int],
    dtype: str,
    fill_value: Any,
    shard_chunk_shape: Sequence[int] | None = None,
) -> ts.TensorStore:
    """Create a resizable zarr v3 array via TensorStore's zarr3 driver.

    ``chunk_shape`` is the read/write granularity (or, when
    ``shard_chunk_shape`` is given, the shard size — many chunks of
    ``shard_chunk_shape`` packed into one file, capping file count for
    small per-step payloads)."""
    metadata: dict[str, Any] = {
        "shape": list(shape),
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunk_shape)}},
        "data_type": dtype,
        "fill_value": fill_value,
    }
    if shard_chunk_shape is not None:
        metadata["codecs"] = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": list(shard_chunk_shape),
                    "codecs": [{"name": "bytes"}],
                },
            }
        ]
    return ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}, "metadata": metadata, "create": True}
    ).result()


def _open_array(path: str) -> ts.TensorStore:
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": path}}).result()


def _resize(store: ts.TensorStore, exclusive_max: Sequence[int]) -> ts.TensorStore:
    """TensorStore resize returns a new store handle (unlike zarr-python's
    in-place ``array.resize``) — callers must reassign it."""
    return store.resize(exclusive_max=list(exclusive_max)).result()


def _fill_value(dtype: str) -> Any:
    return False if dtype == "bool" else 0


@register_backend
class ZarrBackend(StorageBackend):
    name = "zarr"

    def __init__(
        self,
        root: str | Path,
        schema: DatasetSchema,
        field_arrays: dict[str, ts.TensorStore],
        index_arrays: dict[str, ts.TensorStore],
    ):
        self._root = root
        self._schema = schema
        self._field_keys: list[str] = schema.field_keys()
        self._field_arrays = field_arrays
        self._index_arrays = index_arrays
        self._buffers = EpisodeBuffers()
        # In-memory mirror of index/; disk rows are written on finalize/flush.
        self._start: list[int] = []
        self._length: list[int] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._ongoing: list[bool] = []
        self._present: list[set[str]] = []
        self._total = field_arrays[self._field_keys[0]].shape[0] if self._field_keys else 0

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        schema: DatasetSchema,
        path: str | Path | None = None,
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
        field_keys = schema.field_keys()

        field_arrays: dict[str, ts.TensorStore] = {}
        for key in field_keys:
            spec = schema.field(key)
            shape = tuple(spec.shape)
            step_bytes = max(1, np.dtype(spec.dtype).itemsize * math.prod(shape))
            chunk_t = max(1, min(chunk_bytes // step_bytes, _MAX_CHUNK_STEPS))
            outer_chunk = (chunk_t, *shape)
            shard_chunk = None
            if shard_bytes is not None:
                shard_t = chunk_t * max(1, shard_bytes // (chunk_t * step_bytes))
                outer_chunk = (shard_t, *shape)
                shard_chunk = (chunk_t, *shape)
            field_arrays[key] = _create_array(
                os.path.join(path, "data.zarr", "fields", key),
                shape=(0, *shape),
                chunk_shape=outer_chunk,
                shard_chunk_shape=shard_chunk,
                dtype=spec.dtype,
                fill_value=_fill_value(spec.dtype),
            )

        index_arrays: dict[str, ts.TensorStore] = {}
        for name, dtype, fill in _INDEX_FIELDS:
            index_arrays[name] = _create_array(
                os.path.join(path, "data.zarr", "index", name),
                shape=(0,),
                chunk_shape=(_INDEX_CHUNK,),
                dtype=dtype,
                fill_value=fill,
            )
        index_arrays["present"] = _create_array(
            os.path.join(path, "data.zarr", "index", "present"),
            shape=(0, len(field_keys)),
            chunk_shape=(_INDEX_CHUNK, max(1, len(field_keys))),
            dtype="bool",
            fill_value=False,
        )

        backend = cls(path, schema, field_arrays, index_arrays)
        backend._write_manifest()
        return backend

    @classmethod
    def open(cls, path: str | Path) -> "ZarrBackend":
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        if manifest["backend"] != cls.name:
            raise ValueError(
                f"dataset at {path} uses backend {manifest['backend']!r}, not {cls.name!r}"
            )
        schema = DatasetSchema.from_dict(manifest["schema"])
        field_keys = schema.field_keys()
        field_arrays = {
            key: _open_array(os.path.join(path, "data.zarr", "fields", key)) for key in field_keys
        }
        index_names = [name for name, _, _ in _INDEX_FIELDS] + ["present"]
        index_arrays = {
            name: _open_array(os.path.join(path, "data.zarr", "index", name))
            for name in index_names
        }

        backend = cls(path, schema, field_arrays, index_arrays)
        backend._start = index_arrays["start"].read().result().tolist()
        backend._length = index_arrays["length"].read().result().tolist()
        backend._terminated = index_arrays["terminated"].read().result().tolist()
        backend._truncated = index_arrays["truncated"].read().result().tolist()
        backend._ongoing = index_arrays["ongoing"].read().result().tolist()
        present = index_arrays["present"].read().result()
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
        self, field_ids: Sequence[str], selections: Sequence[Selection]
    ) -> Sequence[Mapping[str, np.ndarray]]:
        """Batched read: one fancy-index gather per field across the whole
        batch, instead of one call per selection.

        Ongoing (buffered, not yet finalized) episodes aren't in the flat
        per-field arrays yet, so those selections are read individually
        from the in-memory buffer (already cheap) and merged back in
        original order.
        """
        results: list[dict[str, np.ndarray] | None] = [None] * len(selections)
        finalized_positions: list[int] = []
        for position, selection in enumerate(selections):
            if selection.episode_id in self._buffers:
                results[position] = dict(self._buffers.read(field_ids, selection))
            else:
                finalized_positions.append(position)

        if finalized_positions:
            finalized_selections = [selections[p] for p in finalized_positions]
            lengths = [s.length for s in finalized_selections]
            split_points = np.cumsum(lengths)[:-1]
            abs_rows = np.concatenate(
                [
                    np.arange(
                        self._start[s.episode_id] + s.start, self._start[s.episode_id] + s.stop
                    )
                    for s in finalized_selections
                ]
            )
            for position in finalized_positions:
                results[position] = {}
            for key in field_ids:
                for p, s in zip(finalized_positions, finalized_selections):
                    if key not in self._present[s.episode_id]:
                        raise KeyError(f"episode {s.episode_id} has no field {key!r}")
                store = self._field_arrays[key]
                gathered = store.vindex[ts.array(abs_rows)].read().result()
                pieces = np.split(gathered, split_points, axis=0)
                for position, piece in zip(finalized_positions, pieces):
                    results[position][key] = piece
        return results

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
            store = self._field_arrays[key]
            store = _resize(store, (stop, *store.shape[1:]))
            self._field_arrays[key] = store
            if key in arrays and start < stop:
                store[start:stop].write(arrays[key]).result()
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
        if self._index_arrays["start"].shape[0] >= n:
            return
        for name, _, _ in _INDEX_FIELDS:
            self._index_arrays[name] = _resize(self._index_arrays[name], (n,))
        present = self._index_arrays["present"]
        self._index_arrays["present"] = _resize(present, (n, present.shape[1]))

    def _write_index_row(self, episode_id: int) -> None:
        self._index_arrays["start"][episode_id].write(np.int64(self._start[episode_id])).result()
        self._index_arrays["length"][episode_id].write(
            np.int64(self._length[episode_id])
        ).result()
        self._index_arrays["terminated"][episode_id].write(
            bool(self._terminated[episode_id])
        ).result()
        self._index_arrays["truncated"][episode_id].write(
            bool(self._truncated[episode_id])
        ).result()
        self._index_arrays["ongoing"][episode_id].write(bool(self._ongoing[episode_id])).result()
        present = self._present[episode_id]
        row = np.array([key in present for key in self._field_keys], dtype=bool)
        self._index_arrays["present"][episode_id].write(row).result()

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
