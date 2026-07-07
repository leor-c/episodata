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


@dataclasses.dataclass
class SpaceBlock:
    """Space-oriented payload: several same-space fields stacked along axis 0.

    Backends may return this instead of per-field arrays when a contiguous
    stacked representation is cheaper. The high-level layer normalizes both
    forms into the same observation interface (see :func:`normalize_payload`).
    """

    space: str
    keys: tuple[str, ...]
    data: np.ndarray


#: What backends may return from ``read_fields``.
Payload = Mapping[str, np.ndarray] | Sequence[SpaceBlock]


def normalize_payload(payload: Payload) -> dict[str, np.ndarray]:
    """Normalize field-oriented or space-oriented payloads to field -> array."""
    if isinstance(payload, Mapping):
        return dict(payload)
    out: dict[str, np.ndarray] = {}
    for block in payload:
        for i, key in enumerate(block.keys):
            out[key] = block.data[i]
    return out


class StorageBackend(abc.ABC):
    """Abstract storage backend.

    Concrete backends register themselves via :func:`register_backend` so
    that ``Dataset.open`` can reconstruct the right backend from the
    persisted manifest.
    """

    #: Stable backend identifier persisted in the manifest.
    name: ClassVar[str]

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
        """Persist an updated schema (e.g. after a space rename)."""

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
    def read_fields(self, field_ids: Sequence[str], selection: Selection) -> Payload:
        """Read the given logical fields over a temporal selection.

        Returns arrays shaped ``[selection.length, *space.shape]`` in the
        logical dtype/layout, either field-oriented or as SpaceBlocks.
        """

    # -- writes (online append) ----------------------------------------------

    @abc.abstractmethod
    def create_episode(self) -> int:
        """Start a new (ongoing) episode; returns its id."""

    @abc.abstractmethod
    def append_steps(self, episode_id: int, fields: Mapping[str, np.ndarray]) -> None:
        """Append steps (arrays with a leading time dim) to an ongoing episode."""

    @abc.abstractmethod
    def finalize_episode(self, episode_id: int, terminated: bool, truncated: bool) -> None:
        """Close an ongoing episode and persist its termination flags."""

    def flush(self) -> None:
        """Persist any pending state. Default: no-op."""

    def close(self) -> None:
        self.flush()


_REGISTRY: dict[str, type[StorageBackend]] = {}


def register_backend(cls: type[StorageBackend]) -> type[StorageBackend]:
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str) -> type[StorageBackend]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; registered: {list(_REGISTRY)}"
        ) from None