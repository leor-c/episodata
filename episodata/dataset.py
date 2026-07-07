"""The Dataset: the user-facing entry point.

A dataset is episodes plus a persistent schema. The same logical concepts
and observation API apply whether the data lives in memory, in local files,
or (with a future backend) in object storage.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Callable

import numpy as np

from .backends.base import StorageBackend, get_backend
from .episode import Episode, EpisodeWriter
from .loader import Loader
from .normalize import SEP, normalize_episode, normalize_step, shift_action_out
from .schema import DatasetSchema


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
            schema = DatasetSchema.infer(first)
        dataset = cls.create(schema, path=path, backend=backend, **backend_options)
        dataset.add_episode(first, alignment=alignment)
        for episode in episodes:
            dataset.add_episode(episode, alignment=alignment)
        return dataset

    @classmethod
    def open(cls, path: str, backend: str = "npz_directory") -> "Dataset":
        """Open an existing dataset; the persisted schema is authoritative."""
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

        ``alignment="action_out"`` accepts episodes where row ``t`` holds the
        action taken *at* observation ``t`` (D4RL-style); they are shifted to
        the canonical action-in alignment at write time.
        """
        normalized = normalize_episode(episode)
        if alignment == "action_out":
            normalized = shift_action_out(normalized)
        elif alignment != "action_in":
            raise ValueError(f"unknown alignment {alignment!r}")
        episode_id = self.backend.create_episode()
        self._append_fields(episode_id, normalized.fields)
        return self.end_episode(
            episode_id, terminated=normalized.terminated, truncated=normalized.truncated
        )

    def new_episode(self, initial: Mapping[str, Any] | None = None) -> EpisodeWriter:
        """Start an ongoing episode for online appends.

        ``initial`` may hold an initial step or segment (canonical dict form
        with a leading time dimension).
        """
        writer = EpisodeWriter(self, self.backend.create_episode())
        if initial is not None:
            writer.add_steps(initial)
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

    def add_reset(
        self,
        episode_id: int,
        observations: Mapping[str, Any] | Any,
        infos: Mapping[str, Any] | None = None,
    ) -> None:
        """Write the reset row (row 0) of an episode: the initial observation
        from ``env.reset()`` with dummy zero action/reward.

        Under the action-in convention every subsequent :meth:`add_step` then
        records exactly one ``env.step`` call — the action sent plus the
        observation/reward it produced.
        """
        if self.backend.episode_length(episode_id) > 0:
            raise ValueError("add_reset must write the first step of an episode")
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

    # -- queries -----------------------------------------------------------------

    def loader(
        self,
        fields: list[str] | None = None,
        batch_size: int = 1,
        sequence_length: int | None = None,
        context_length: int | None = None,
        target_length: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        filter: Callable[[Episode], bool] | None = None,
    ) -> Loader:
        """Build a segment loader. See :class:`Loader`."""
        return Loader(
            self,
            fields=fields,
            batch_size=batch_size,
            sequence_length=sequence_length,
            context_length=context_length,
            target_length=target_length,
            shuffle=shuffle,
            seed=seed,
            filter=filter,
        )

    def sample_transitions(
        self,
        batch_size: int,
        fields: list[str] | None = None,
        seed: int | None = None,
        filter: Callable[[Episode], bool] | None = None,
    ):
        """One-shot transition sampling; see :meth:`Loader.sample_transitions`."""
        loader = self.loader(
            fields=fields, batch_size=batch_size, sequence_length=2, seed=seed, filter=filter
        )
        return loader.sample_transitions()

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