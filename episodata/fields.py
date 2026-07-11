"""Generic schema-backed field containers.

Observations, actions, rewards and infos are all the same thing
structurally: named arrays with hierarchical access. :class:`Fields` is the
generic container — flat storage, with groups (from ``/`` in field keys)
reconstructed on access. Arrays may carry arbitrary leading dims (single
step, ``[T, ...]`` segment, or ``[B, T, ...]`` batch) — the grouping logic
is the same.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np

from .schema import SEP

if TYPE_CHECKING:
    from .schema import DatasetSchema


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


class Fields(Mapping):
    """Flat named arrays of one role, with field and group access.

    Segments hand one of these out per role (``seg.observation``,
    ``seg.action``, ...). Access is explicit — a name is a field or a group,
    nothing else::

        obs["front_camera"]           # flat field access
        obs.front_camera              # field attribute access
        obs["inventory/stone"]        # flat path keys always work
        obs.inventory.stone           # group attribute access
        for key, value in obs.inventory.items(): ...

    Method names (``schema`` and the Mapping methods ``keys`` / ``items`` /
    ``values`` / ``get``) win attribute lookup over a same-named field;
    brackets always reach the field.
    """

    def __init__(self, data: Mapping[str, np.ndarray], schema: "DatasetSchema"):
        self._data = dict(data)
        self._schema = schema

    @property
    def schema(self) -> "DatasetSchema":
        return self._schema

    def _resolve(self, name: str):
        """Single source of truth for name resolution: field, then group."""
        if name in self._data:
            return self._data[name]
        if self._is_group(name):
            return FieldGroup(name, self._data, self._schema)
        raise KeyError(name)

    def __getitem__(self, key: str) -> np.ndarray:
        return self._resolve(key)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._resolve(name)
        except KeyError:
            raise AttributeError(
                f"no field or group named {name!r}; fields: {list(self._data)}"
            ) from None

    def _is_group(self, name: str) -> bool:
        head = f"{name}{SEP}"
        return any(k.startswith(head) for k in self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        shapes = {k: tuple(v.shape) for k, v in self._data.items()}
        return f"{type(self).__name__}({shapes})"


def role_view(
    data: Mapping[str, np.ndarray], schema: "DatasetSchema", role: str
) -> np.ndarray | Fields:
    """Build the access object for one role's fields.

    A role whose only present field carries the role's own name — the flat
    key that normalization gives a bare (non-dict) source — unwraps straight
    to that array; anything else (including an empty role) is a
    :class:`Fields` view.
    """
    present = {k: v for k, v in data.items() if schema.fields[k].role == role}
    if set(present) == {role}:
        return present[role]
    return Fields(present, schema)
