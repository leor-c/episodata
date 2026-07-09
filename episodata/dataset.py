"""The Dataset: the user-facing entry point.

A dataset is episodes plus a persistent schema. The same logical concepts
and observation API apply whether the data lives in memory, in local files,
or (with a future backend) in object storage.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Callable

import numpy as np

from .backends.base import Selection, StorageBackend, get_backend, normalize_payload
from .episode import Episode, EpisodeWriter
from .sampling import SegmentDataset, SegmentStream
from .normalize import SEP, normalize_episode, normalize_full_episode, normalize_step, shift_action_out
from .schema import DatasetSchema
from .vector import VectorWriter


class Dataset:
    def __init__(self, backend: StorageBackend):
        self.backend = backend

    # -- construction ------------------------------------------------------

    @classmethod
    def create(
        cls,
        schema: DatasetSchema,
        path: str | None = None,
        backend: str | None = None,
        **backend_options: Any,
    ) -> "Dataset":
        """Create an empty dataset with a declared schema.

        ``backend`` defaults to "memory" without a path and "npz_directory"
        with one.
        """
        if backend is None:
            backend = "memory" if path is None else "npz_directory"
        backend_cls = get_backend(backend)
        return cls(backend_cls.create(schema, path=path, **backend_options))

    @classmethod
    def from_episodes(
        cls,
        episodes: Iterable[Mapping[str, Any]],
        schema: DatasetSchema | None = None,
        path: str | None = None,
        backend: str | None = None,
        alignment: str = "action_in",
        **backend_options: Any,
    ) -> "Dataset":
        """Build a dataset from canonical episode dicts.

        Without ``schema``, the schema is inferred from the first episode
        (automatic mode) and becomes authoritative once persisted.
        ``alignment`` is forwarded to :meth:`add_episode`.
        """
        episodes = iter(episodes)
        first = next(episodes, None)
        if first is None:
            raise ValueError("from_episodes requires at least one episode; use Dataset.create for an empty dataset")
        if schema is None:
            schema = DatasetSchema.infer(first, alignment=alignment)
        dataset = cls.create(schema, path=path, backend=backend, **backend_options)
        dataset.add_episode(first, alignment=alignment)
        for episode in episodes:
            dataset.add_episode(episode, alignment=alignment)
        return dataset

    @classmethod
    def open(cls, path: str, backend: str | None = None) -> "Dataset":
        """Open an existing dataset; the persisted schema is authoritative.

        The persisted manifest names the backend, so ``backend`` is only
        needed for storage without a ``manifest.json``.
        """
        if backend is None:
            with open(os.path.join(path, "manifest.json")) as f:
                backend = json.load(f)["backend"]
        return cls(get_backend(backend).open(path))

    # -- core accessors ------------------------------------------------------

    @property
    def schema(self) -> DatasetSchema:
        return self.backend.schema

    @property
    def num_episodes(self) -> int:
        return self.backend.num_episodes

    def episode(self, episode_id: int) -> Episode:
        if not 0 <= episode_id < self.num_episodes:
            raise IndexError(f"episode {episode_id} out of range [0, {self.num_episodes})")
        return Episode(self, episode_id)

    def episodes(self) -> Iterator[Episode]:
        for i in range(self.num_episodes):
            yield self.episode(i)

    def __len__(self) -> int:
        return self.num_episodes

    # -- writes ----------------------------------------------------------------

    def add_episode(self, episode: Mapping[str, Any], alignment: str = "action_in") -> Episode:
        """Add one complete episode from a canonical episode dict.

        Under the default ``alignment="action_in"``, observations carry one
        entry more than actions/rewards (the reset row plus one entry per
        step); the reset row's dummy zero action/reward is synthesized here,
        mirroring ``new_episode``. ``alignment="action_out"`` instead accepts
        episodes where row ``t`` holds the action taken *at* observation ``t``
        (D4RL-style, all fields equal length); they are shifted to the
        canonical action-in alignment at write time.
        """
        if alignment == "action_in":
            normalized = normalize_full_episode(episode)
        elif alignment == "action_out":
            normalized = shift_action_out(normalize_episode(episode))
        else:
            raise ValueError(f"unknown alignment {alignment!r}")
        episode_id = self.backend.create_episode()
        self._append_fields(episode_id, normalized.fields)
        return self.end_episode(
            episode_id, terminated=normalized.terminated, truncated=normalized.truncated
        )

    def new_episode(
        self,
        observations: Mapping[str, Any] | Any = None,
        infos: Mapping[str, Any] | None = None,
    ) -> EpisodeWriter:
        """Start an ongoing episode for online appends.

        An episode begins at reset: pass what ``env.reset()`` returned and
        row 0 is written — the initial observation with dummy zero
        action/reward. Under the action-in convention every subsequent
        :meth:`add_step` then records exactly one ``env.step`` call. The
        bare form starts an empty episode for flows that append complete
        rows via :meth:`add_steps` (including a pre-zeroed reset row).
        """
        if observations is None and infos is not None:
            raise ValueError("new_episode: infos requires observations")
        writer = EpisodeWriter(self, self.backend.create_episode())
        if observations is not None:
            self._write_reset_row(writer.episode_id, observations, infos)
        return writer

    def resume_episode(self, episode_id: int) -> EpisodeWriter:
        """Reattach a writer to an ongoing episode.

        The writer is a stateless handle; only ``episode_id`` needs to be
        kept (or persisted) to continue an episode later — including after
        ``Dataset.open`` on a persistent backend.
        """
        if not self.backend.episode_ongoing(episode_id):
            raise ValueError(f"episode {episode_id} is finalized")
        return EpisodeWriter(self, episode_id)

    def vector_writer(self, num_envs: int | None = None) -> VectorWriter:
        """Collect from N parallel environments with staggered episode
        boundaries; see :class:`VectorWriter`.

        ``num_envs`` may be omitted and is then inferred from the first
        :meth:`VectorWriter.reset` call.
        """
        return VectorWriter(self, num_envs=num_envs)

    def _write_reset_row(
        self,
        episode_id: int,
        observations: Mapping[str, Any] | Any,
        infos: Mapping[str, Any] | None = None,
    ) -> None:
        """Write the reset row (row 0): the initial observation from
        ``env.reset()`` with dummy zero action/reward."""
        if self.backend.episode_length(episode_id) > 0:
            raise ValueError("the reset row must be the first row of an episode")
        step: dict[str, Any] = {"observations": observations}
        if infos is not None:
            step["infos"] = infos
        fields = normalize_step(step).fields
        for key, spec in self.schema.fields.items():
            if key in fields or spec.optional or spec.role == "observation":
                continue
            space = self.schema.space_of(key)
            fields[key] = np.zeros((1, *space.shape), dtype=space.dtype)
        self._append_fields(episode_id, fields)

    def add_step(self, episode_id: int, step: Mapping[str, Any]) -> None:
        """Append one step (no time dimension) to an ongoing episode.

        The step may carry the Gymnasium ``terminated`` / ``truncated``
        signals returned by ``env.step``; a True signal finalizes the
        episode, exactly as it ends the Gymnasium episode.
        """
        normalized = normalize_step(step)
        self._append_fields(episode_id, normalized.fields)
        if normalized.terminated or normalized.truncated:
            self.end_episode(episode_id, normalized.terminated, normalized.truncated)

    def add_steps(self, episode_id: int, steps: Mapping[str, Any]) -> None:
        """Append a segment (leading time dimension) to an ongoing episode.

        Per-step ``terminated`` / ``truncated`` arrays are accepted; they may
        only be True on the segment's final step, which then finalizes the
        episode.
        """
        normalized = normalize_episode(steps)
        self._append_fields(episode_id, normalized.fields)
        if normalized.terminated or normalized.truncated:
            self.end_episode(episode_id, normalized.terminated, normalized.truncated)

    def end_episode(
        self, episode_id: int, terminated: bool = False, truncated: bool = False
    ) -> Episode:
        """Finalize an ongoing episode and persist its termination flags."""
        if not self.backend.episode_ongoing(episode_id):
            raise ValueError(f"episode {episode_id} is already finalized")
        self.backend.finalize_episode(episode_id, terminated, truncated)
        return Episode(self, episode_id)

    def _append_fields(self, episode_id: int, fields: Mapping[str, np.ndarray]) -> None:
        if not self.backend.episode_ongoing(episode_id):
            raise ValueError(f"episode {episode_id} is finalized")
        self.backend.append_steps(episode_id, self._validate_fields(fields))

    def _add_step_batch(
        self, episode_ids: Sequence[int], fields: Mapping[str, np.ndarray]
    ) -> None:
        """Append one already-normalized step per episode (row ``i`` goes to
        ``episode_ids[i]``)."""
        fields = self._validate_fields(fields)
        for key, arr in fields.items():
            if len(arr) != len(episode_ids):
                raise ValueError(
                    f"field {key!r} has {len(arr)} rows for {len(episode_ids)} episodes"
                )
        for episode_id in episode_ids:
            if not self.backend.episode_ongoing(episode_id):
                raise ValueError(f"episode {episode_id} is finalized")
        self.backend.append_steps_batch(list(episode_ids), fields)

    def rename_space(self, old: str, new: str) -> None:
        """Hybrid mode: rename an (inferred) space and persist the schema."""
        schema = self.schema
        schema.rename_space(old, new)
        self.backend.write_schema(schema)

    def rename_field(self, old: str, new: str) -> None:
        raise NotImplementedError(
            "renaming a field changes its stable storage identifier; "
            "not supported by the v1 backends"
        )

    def flush(self) -> None:
        self.backend.flush()

    def close(self) -> None:
        self.backend.close()

    def copy_to(
        self,
        path: str | None = None,
        backend: str | None = None,
        **backend_options: Any,
    ) -> "Dataset":
        """Copy this dataset into a new one on another backend or path.

        Streams one episode at a time through the storage boundary — use it
        to migrate a dataset that outgrew its backend (e.g. ``npz_directory``
        to ``zarr``). Ongoing episodes are copied ongoing.
        """
        destination = Dataset.create(self.schema, path=path, backend=backend, **backend_options)
        keys = self.schema.field_keys()
        for episode_id in range(self.num_episodes):
            destination_id = destination.backend.create_episode()
            length = self.backend.episode_length(episode_id)
            if length:
                selection = Selection(episode_id, 0, length)
                fields: dict[str, np.ndarray] = {}
                for key in keys:
                    try:
                        payload = self.backend.read_fields([key], selection)
                    except KeyError:  # field absent from this episode
                        continue
                    fields.update(normalize_payload(payload))
                destination.backend.append_steps(destination_id, fields)
            if not self.backend.episode_ongoing(episode_id):
                destination.backend.finalize_episode(
                    destination_id,
                    terminated=self.backend.episode_terminated(episode_id),
                    truncated=self.backend.episode_truncated(episode_id),
                )
        destination.flush()
        return destination

    # -- queries -----------------------------------------------------------------

    def segment_stream(
        self,
        fields: list[str] | None = None,
        batch_size: int = 1,
        sequence_length: int | None = None,
        context_length: int | None = None,
        target_length: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        filter: Callable[[Episode], bool] | None = None,
        pad: str | None = "suffix",
    ) -> SegmentStream:
        """Build a segment stream. See :class:`SegmentStream`.

        ``pad`` controls segments drawn from episodes shorter than the
        requested length: zero-padded at the end (``"suffix"``, default) or
        at the start (``"prefix"``), with ``Batch.mask`` marking real
        steps; ``None`` skips short episodes.
        """
        return SegmentStream(
            self,
            fields=fields,
            batch_size=batch_size,
            sequence_length=sequence_length,
            context_length=context_length,
            target_length=target_length,
            shuffle=shuffle,
            seed=seed,
            filter=filter,
            pad=pad,
        )

    def segments(
        self,
        fields: list[str] | None = None,
        sequence_length: int | None = None,
        context_length: int | None = None,
        target_length: int | None = None,
        filter: Callable[[Episode], bool] | None = None,
        pad: str | None = "suffix",
    ) -> SegmentDataset:
        """Build a map-style, indexable view over segments. See
        :class:`SegmentDataset` — suited to ``torch.utils.data.DataLoader``
        and its ``num_workers`` parallelism, unlike :meth:`segment_stream`.
        ``pad`` behaves as in :meth:`segment_stream`. On a growing dataset, call
        :meth:`SegmentDataset.refresh` between epochs to make newly
        appended episodes visible."""
        return SegmentDataset(
            self,
            fields=fields,
            sequence_length=sequence_length,
            context_length=context_length,
            target_length=target_length,
            filter=filter,
            pad=pad,
        )

    def sample_transitions(
        self,
        batch_size: int,
        fields: list[str] | None = None,
        seed: int | None = None,
        filter: Callable[[Episode], bool] | None = None,
    ):
        """One-shot transition sampling; see :meth:`SegmentStream.sample_transitions`.

        Sampled without padding: a padded segment would fabricate a
        transition into a zero-filled next observation.
        """
        stream = self.segment_stream(
            fields=fields, batch_size=batch_size, sequence_length=2, seed=seed, filter=filter,
            pad=None,
        )
        return stream.sample_transitions()

    # -- internal helpers ------------------------------------------------------------

    def _resolve_fields(self, fields: list[str] | None) -> list[str]:
        """Resolve requested names to schema field keys.

        A name that is a path prefix (e.g. ``"keyboard"``) selects every
        field beneath it.
        """
        if fields is None:
            return self.schema.field_keys()
        resolved: list[str] = []
        for name in fields:
            if name in self.schema.fields:
                resolved.append(name)
                continue
            head = f"{name}{SEP}"
            members = [k for k in self.schema.fields if k.startswith(head)]
            if not members:
                raise KeyError(
                    f"unknown field {name!r}; schema has {list(self.schema.fields)}"
                )
            resolved.extend(members)
        return resolved

    def _validate_fields(self, fields: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Validate appended data against the schema and cast to logical dtype."""
        schema = self.schema
        out: dict[str, np.ndarray] = {}
        for key, arr in fields.items():
            if key not in schema.fields:
                raise KeyError(f"field {key!r} is not in the schema")
            space = schema.space_of(key)
            arr = np.asarray(arr)
            expected = (len(arr), *space.shape)
            if tuple(arr.shape) != expected:
                raise ValueError(
                    f"field {key!r}: got shape {tuple(arr.shape)}, expected {expected} "
                    f"(space {space.key!r})"
                )
            out[key] = arr.astype(space.dtype, copy=False)
        missing = [
            k for k, f in schema.fields.items() if k not in out and not f.optional
        ]
        if missing:
            raise ValueError(f"missing required fields {missing}")
        return out

    def __repr__(self) -> str:
        return (
            f"Dataset(backend={self.backend.name!r}, num_episodes={self.num_episodes}, "
            f"fields={list(self.schema.fields)})"
        )