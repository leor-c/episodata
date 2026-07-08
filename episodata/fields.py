"""Generic schema-backed field containers.

Observations, actions, rewards and infos are all the same thing
structurally: named arrays with hierarchical access. :class:`Fields` is the
generic container — flat storage, with spaces (from the schema) and groups
(from ``/`` in field keys) reconstructed on access. Arrays may carry
arbitrary leading dims (single step, ``[T, ...]`` segment, or ``[B, T, ...]``
batch) — the grouping logic is the same.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np

from .normalize import SEP

if TYPE_CHECKING:
    from .schema import DatasetSchema, SpaceSpec


class SpaceView(Mapping):
    """Read-only view of the fields of one space present in a container."""

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


class Fields(Mapping):
    """Flat named arrays with schema-driven space, group and role access.

    Access patterns::

        fields["front_camera"]          # flat field access
        fields.image.front_camera       # space attribute access
        fields.image["front_camera"]
        for key, value in fields.image.items(): ...

    Fields with path keys (``"keyboard/w"``) additionally form groups::

        fields["keyboard/w"]            # flat access always works
        fields.keyboard.w               # group attribute access
        for key, value in fields.keyboard.items(): ...

    Schema roles give semantic sub-views holding just the fields of one
    role::

        fields.observations             # observation-role fields
        fields.actions                  # action-role fields
        fields.rewards                  # reward-role fields

    Name resolution order for attributes and keys: role view, space, group,
    field.
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

    def select(self, fields: list[str]) -> "Fields":
        return type(self)({k: self._data[k] for k in fields}, self._schema)

    # -- role views ---------------------------------------------------------

    def _role_view(self, role: str) -> "Fields":
        return self.select(
            [k for k in self._data if self._schema.fields[k].role == role]
        )

    @property
    def observations(self) -> "Fields":
        """Sub-view holding only the observation-role fields."""
        return self._role_view("observation")

    @property
    def actions(self) -> "Fields":
        """Sub-view holding only the action-role fields."""
        return self._role_view("action")

    @property
    def rewards(self) -> "Fields":
        """Sub-view holding only the reward-role fields."""
        return self._role_view("reward")

    @property
    def infos(self) -> "Fields":
        """Sub-view holding only the info-role fields."""
        return self._role_view("info")

    def __repr__(self) -> str:
        shapes = {k: tuple(v.shape) for k, v in self._data.items()}
        return f"{type(self).__name__}({shapes})"


#: Backward-compatible name: an observation is just fields, like everything
#: else. Prefer :class:`Fields` in new code.
Observation = Fields
