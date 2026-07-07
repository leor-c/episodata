"""Schema-backed observation structure.

An :class:`Observation` is a collection of named arrays grouped into spaces
by the schema. The internal representation is flat; the hierarchy is
reconstructed from the schema. Arrays may carry arbitrary leading dims
(single step, ``[T, ...]`` segment, or ``[B, T, ...]`` batch) — the grouping
logic is the same.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np

from .normalize import SEP

if TYPE_CHECKING:
    from .schema import DatasetSchema, SpaceSpec


class SpaceView(Mapping):
    """Read-only view of the fields of one space present in an observation."""

    def __init__(self, space: "SpaceSpec", data: dict[str, np.ndarray]):
        self._space = space
        self._data = data

    @property
    def spec(self) -> "SpaceSpec":
        return self._space

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"space {self._space.key!r} has no field {name!r}; "
                f"available: {list(self._data)}"
            ) from None

    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def stacked(self, axis: int = 0) -> np.ndarray:
        """Stack all fields of this space along a new axis.

        Valid because fields in one space share shape and dtype.
        """
        return np.stack(list(self._data.values()), axis=axis)

    def __repr__(self) -> str:
        return f"SpaceView({self._space.key!r}, fields={list(self._data)})"


class FieldGroup(Mapping):
    """Hierarchical view over fields sharing a path prefix.

    The hierarchy comes from ``/`` in flat field keys (``"keyboard/w"``);
    storage and field identifiers stay flat. Members are the direct
    children: leaf arrays or nested groups.
    """

    def __init__(self, prefix: str, data: dict[str, np.ndarray], schema: "DatasetSchema"):
        self._prefix = prefix
        self._data = data
        self._schema = schema

    def _resolve(self, name: str):
        full = f"{self._prefix}{SEP}{name}"
        if full in self._data:
            return self._data[full]
        head = f"{full}{SEP}"
        if any(k.startswith(head) for k in self._data):
            return FieldGroup(full, self._data, self._schema)
        raise KeyError(full)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._resolve(name)
        except KeyError:
            raise AttributeError(
                f"group {self._prefix!r} has no member {name!r}; "
                f"members: {list(self)}"
            ) from None

    def __getitem__(self, key: str):
        """Resolve a member name or a relative path below this prefix."""
        return self._resolve(key)

    def __iter__(self) -> Iterator[str]:
        seen: dict[str, None] = {}
        head = f"{self._prefix}{SEP}"
        for key in self._data:
            if key.startswith(head):
                seen.setdefault(key[len(head):].split(SEP, 1)[0])
        return iter(seen)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return f"FieldGroup({self._prefix!r}, members={list(self)})"


class Observation(Mapping):
    """Flat named arrays with schema-driven space access.

    Access patterns::

        obs["front_camera"]          # flat field access
        obs.image.front_camera       # space attribute access
        obs.image["front_camera"]
        for key, value in obs.image.items(): ...

    Fields with path keys (``"keyboard/w"``) additionally form groups::

        obs["keyboard/w"]            # flat access always works
        obs.keyboard.w               # group attribute access
        for key, value in obs.keyboard.items(): ...

    Name resolution order for attributes and keys: space, group, field.
    """

    def __init__(self, data: Mapping[str, np.ndarray], schema: "DatasetSchema"):
        self._data = dict(data)
        self._schema = schema

    @property
    def schema(self) -> "DatasetSchema":
        return self._schema

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._data:
            return self._data[key]
        if key in self._schema.spaces:
            return self._space_view(key)
        if self._is_group(key):
            return FieldGroup(key, self._data, self._schema)
        raise KeyError(key)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._schema.spaces:
            return self._space_view(name)
        if self._is_group(name):
            return FieldGroup(name, self._data, self._schema)
        if name in self._data:
            return self._data[name]
        raise AttributeError(
            f"no field, group or space named {name!r}; fields: {list(self._data)}, "
            f"spaces: {list(self._schema.spaces)}"
        )

    def _is_group(self, name: str) -> bool:
        head = f"{name}{SEP}"
        return any(k.startswith(head) for k in self._data)

    def _space_view(self, space_key: str) -> SpaceView:
        members = {
            k: self._data[k]
            for k in self._schema.fields_in_space(space_key)
            if k in self._data
        }
        return SpaceView(self._schema.space(space_key), members)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def select(self, fields: list[str]) -> "Observation":
        return Observation({k: self._data[k] for k in fields}, self._schema)

    def __repr__(self) -> str:
        shapes = {k: tuple(v.shape) for k, v in self._data.items()}
        return f"{type(self).__name__}({shapes})"


class Batch(Observation):
    """A batched window of fields with arrays shaped ``[B, L, ...]``.

    When the loader was configured with context/target windows, ``context``
    and ``target`` expose the corresponding time slices.
    """

    def __init__(
        self,
        data: Mapping[str, np.ndarray],
        schema: "DatasetSchema",
        context_length: int | None = None,
        target_length: int | None = None,
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ):
        super().__init__(data, schema)
        self._context_length = context_length
        self._target_length = target_length
        #: Per-step flags [B, L]: True only on the final step of a
        #: terminated (resp. truncated) episode.
        self.terminated = terminated
        self.truncated = truncated
        #: Per-step validity [B, L]: True on real steps, False on the
        #: zero-padding of a window drawn from a too-short episode.
        self.mask = mask

    @property
    def context(self) -> "Batch":
        if not self._context_length:
            raise ValueError("loader was not configured with context_length")
        return self._time_slice(0, self._context_length)

    @property
    def target(self) -> "Batch":
        if self._target_length is None:
            raise ValueError("loader was not configured with target_length")
        start = self._context_length or 0
        return self._time_slice(start, start + self._target_length)

    def _time_slice(self, start: int, stop: int) -> "Batch":
        return Batch(
            {k: v[:, start:stop] for k, v in self._data.items()},
            self._schema,
            terminated=None if self.terminated is None else self.terminated[:, start:stop],
            truncated=None if self.truncated is None else self.truncated[:, start:stop],
            mask=None if self.mask is None else self.mask[:, start:stop],
        )