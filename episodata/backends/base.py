"""The storage boundary.

A backend operates on stable logical field identifiers and temporal
selections. It owns the physical side entirely: file layout, shards,
codecs, chunking, indexing, caching, decoding. It must expose the logical
representation declared by the schema regardless of how data is physically
stored (e.g. a video codec decoded back to the logical dtype/layout).

The high-level layer owns schema interpretation, observation structure,
queries, sampling and batch construction. Backends never implement
Python-level access such as ``obs.image.front``.
"""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from ..schema import DatasetSchema


@dataclasses.dataclass(frozen=True)
class Selection:
    """A temporal selection: steps ``[start, stop)`` of one episode."""

    episode_id: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


class StorageBackend(abc.ABC):
    """Abstract storage backend.

    Concrete backends register themselves via :func:`register_backend` so
    that ``Dataset.open`` can reconstruct the right backend from the
    persisted manifest.
    """

    #: Stable backend identifier persisted in the manifest.
    name: ClassVar[str]

    _revision: int = 0

    @property
    def revision(self) -> int:
        """Monotonic counter of episode mutations (create/append/finalize).

        Consumers snapshotting derived state (e.g. a segment index) compare
        revisions to skip rebuilding when nothing changed.
        """
        return self._revision

    def _touch(self) -> None:
        """Record an episode mutation. Concrete backends call this from
        ``create_episode``, ``append_steps`` and ``finalize_episode``."""
        self._revision += 1

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def create(cls, schema: DatasetSchema, path: str | None = None, **options: Any) -> "StorageBackend":
        """Create empty storage for a new dataset."""

    @classmethod
    @abc.abstractmethod
    def open(cls, path: str) -> "StorageBackend":
        """Open existing storage; the persisted schema is authoritative."""

    @property
    @abc.abstractmethod
    def schema(self) -> DatasetSchema:
        """The persisted logical schema."""

    @abc.abstractmethod
    def write_schema(self, schema: DatasetSchema) -> None:
        """Persist an updated schema (e.g. after a field rename)."""

    # -- episode index -------------------------------------------------------

    @property
    @abc.abstractmethod
    def num_episodes(self) -> int: ...

    @abc.abstractmethod
    def episode_length(self, episode_id: int) -> int: ...

    @abc.abstractmethod
    def episode_terminated(self, episode_id: int) -> bool: ...

    @abc.abstractmethod
    def episode_truncated(self, episode_id: int) -> bool: ...

    @abc.abstractmethod
    def episode_ongoing(self, episode_id: int) -> bool:
        """True while an episode is still open for appends."""

    # -- reads ---------------------------------------------------------------

    @abc.abstractmethod
    def read_fields(
        self, field_ids: Sequence[str], selection: Selection
    ) -> Mapping[str, np.ndarray]:
        """Read the given logical fields over a temporal selection.

        Returns arrays shaped ``[selection.length, *field.shape]`` in the
        logical dtype/layout, keyed by field id.
        """

    # -- writes (online append) ----------------------------------------------

    @abc.abstractmethod
    def create_episode(self) -> int:
        """Start a new (ongoing) episode; returns its id."""

    @abc.abstractmethod
    def append_steps(self, episode_id: int, fields: Mapping[str, np.ndarray]) -> None:
        """Append steps (arrays with a leading time dim) to an ongoing episode."""

    def append_steps_batch(
        self, episode_ids: Sequence[int], fields: Mapping[str, np.ndarray]
    ) -> None:
        """Append one step to each of several ongoing episodes.

        Arrays are shaped ``[N, *field.shape]``; row ``i`` goes to
        ``episode_ids[i]``. The default implementation loops the
        per-episode append; backends may override with a vectorized write
        (an override must call ``_touch()`` at least once).
        """
        for i, episode_id in enumerate(episode_ids):
            self.append_steps(episode_id, {k: a[i : i + 1] for k, a in fields.items()})

    @abc.abstractmethod
    def finalize_episode(self, episode_id: int, terminated: bool, truncated: bool) -> None:
        """Close an ongoing episode and persist its termination flags."""

    def flush(self) -> None:
        """Persist any pending state. Default: no-op."""

    def close(self) -> None:
        self.flush()


_REGISTRY: dict[str, type[StorageBackend]] = {}
_MISSING: dict[str, str] = {}


def register_backend(cls: type[StorageBackend]) -> type[StorageBackend]:
    _REGISTRY[cls.name] = cls
    return cls


def register_missing_backend(name: str, reason: str) -> None:
    """Record why an optional backend is unavailable, for a clear error."""
    _MISSING[name] = reason


def get_backend(name: str) -> type[StorageBackend]:
    try:
        return _REGISTRY[name]
    except KeyError:
        if name in _MISSING:
            raise ImportError(f"backend {name!r} is unavailable: {_MISSING[name]}") from None
        raise KeyError(
            f"unknown backend {name!r}; registered: {list(_REGISTRY)}"
        ) from None