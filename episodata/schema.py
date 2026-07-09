"""Logical schema: spaces, fields, and the dataset schema.

The schema defines what the data *means*. It is independent of how any
backend physically stores the data. Once a dataset is created, the persisted
schema is authoritative; it is never re-inferred on reopen.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

#: Field roles. Observations, actions and rewards are temporal fields that
#: exist at every step of an episode. "info" covers auxiliary per-step data.
ROLES = ("observation", "action", "reward", "info")


@dataclasses.dataclass
class SpaceSpec:
    """Shared logical format of a set of fields.

    Fields in the same space share shape/dtype guarantees, which enables
    structural operations such as stacking, validation and generic
    transforms.
    """

    key: str
    shape: tuple[int, ...]
    dtype: str
    low: float | None = None
    high: float | None = None
    layout: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shape = tuple(int(s) for s in self.shape)
        self.dtype = np.dtype(self.dtype).name

    def matches(self, per_step_array: np.ndarray) -> bool:
        """Whether a per-step value structurally belongs to this space."""
        arr = np.asarray(per_step_array)
        return tuple(arr.shape) == self.shape and arr.dtype == np.dtype(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SpaceSpec:
        return cls(
            key=d["key"],
            shape=tuple(d["shape"]),
            dtype=d["dtype"],
            low=d.get("low"),
            high=d.get("high"),
            layout=d.get("layout"),
            metadata=dict(d.get("metadata", {})),
        )


@dataclasses.dataclass
class FieldSpec:
    """A named field with a stable identity, belonging to exactly one space.

    The field key is the stable logical identifier used across the storage
    boundary. Space membership (structure) and semantic meaning are separate
    concepts.
    """

    key: str
    space: str
    role: str = "observation"
    semantic_type: str | None = None
    optional: bool = False
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}, expected one of {ROLES}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FieldSpec:
        return cls(
            key=d["key"],
            space=d["space"],
            role=d.get("role", "observation"),
            semantic_type=d.get("semantic_type"),
            optional=d.get("optional", False),
            metadata=dict(d.get("metadata", {})),
        )


class DatasetSchema:
    """The persistent logical schema of a dataset."""

    def __init__(self, spaces: Iterable[SpaceSpec], fields: Iterable[FieldSpec]):
        self.spaces: dict[str, SpaceSpec] = {}
        self.fields: dict[str, FieldSpec] = {}
        for space in spaces:
            if space.key in self.spaces:
                raise ValueError(f"duplicate space key {space.key!r}")
            self.spaces[space.key] = space
        for field in fields:
            if field.key in self.fields:
                raise ValueError(f"duplicate field key {field.key!r}")
            if field.space not in self.spaces:
                raise ValueError(
                    f"field {field.key!r} references unknown space {field.space!r}"
                )
            self.fields[field.key] = field
        for key in self.fields:
            # A field named after a space is fine when it is that space's
            # only field (attribute access unwraps the trivial space to the
            # array). With siblings present the space wins attribute lookup
            # and shadows the field, so flag the ambiguity.
            if key in self.spaces and self.fields_in_space(key) != [key]:
                warnings.warn(
                    f"field {key!r} is shadowed by space {key!r} for attribute "
                    f"access; use fields[{key!r}] to read the field",
                    stacklevel=2,
                )

    # -- lookup ----------------------------------------------------------

    def field(self, key: str) -> FieldSpec:
        return self.fields[key]

    def space(self, key: str) -> SpaceSpec:
        return self.spaces[key]

    def space_of(self, field_key: str) -> SpaceSpec:
        return self.spaces[self.fields[field_key].space]

    def field_keys(self, role: str | None = None) -> list[str]:
        if role is None:
            return list(self.fields)
        return [k for k, f in self.fields.items() if f.role == role]

    def fields_in_space(self, space_key: str) -> list[str]:
        return [k for k, f in self.fields.items() if f.space == space_key]

    # -- refinement (hybrid mode) -----------------------------------------

    def rename_space(self, old: str, new: str) -> None:
        if new in self.spaces:
            raise ValueError(f"space key {new!r} already exists")
        space = self.spaces.pop(old)
        space.key = new
        self.spaces[new] = space
        for field in self.fields.values():
            if field.space == old:
                field.space = new

    def rename_field(self, old: str, new: str) -> None:
        if new in self.fields:
            raise ValueError(f"field key {new!r} already exists")
        field = self.fields.pop(old)
        field.key = new
        self.fields[new] = field

    # -- validation --------------------------------------------------------

    def validate_step_value(self, field_key: str, per_step_array: np.ndarray) -> None:
        space = self.space_of(field_key)
        arr = np.asarray(per_step_array)
        if tuple(arr.shape) != space.shape:
            raise ValueError(
                f"field {field_key!r}: per-step shape {tuple(arr.shape)} does not "
                f"match space {space.key!r} shape {space.shape}"
            )

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "spaces": [s.to_dict() for s in self.spaces.values()],
            "fields": [f.to_dict() for f in self.fields.values()],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DatasetSchema:
        return cls(
            spaces=[SpaceSpec.from_dict(s) for s in d["spaces"]],
            fields=[FieldSpec.from_dict(f) for f in d["fields"]],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> DatasetSchema:
        return cls.from_dict(json.loads(s))

    # -- inference (automatic mode) -----------------------------------------

    @classmethod
    def infer(
        cls, example_episode: Mapping[str, Any], alignment: str | None = None
    ) -> DatasetSchema:
        """Infer a schema from one example episode (canonical episode dict).

        Shape and dtype are inferred reliably. Space keys are generated from
        structural heuristics and can be renamed later (hybrid mode). The
        episode's alignment is read from its boundary keys
        (``initial_observation`` marks action-in, ``final_observation``
        action-out); ``alignment`` is only needed for action-out data
        without its final observation (see ``resolve_alignment``).
        """
        from .normalize import (
            normalize_action_out_episode,
            normalize_full_episode,
            resolve_alignment,
        )

        alignment = resolve_alignment(example_episode, alignment)
        if alignment == "action_in":
            normalized = normalize_full_episode(example_episode)
        else:
            normalized = normalize_action_out_episode(example_episode)
        spaces: dict[str, SpaceSpec] = {}
        fields: list[FieldSpec] = []
        for key, array in normalized.fields.items():
            role = normalized.roles[key]
            per_step = array[0]
            spec = _infer_space(role, per_step)
            space_key = _assign_space(spaces, spec)
            fields.append(FieldSpec(key=key, space=space_key, role=role, optional=(role == "info")))
        _collapse_single_action_space(spaces, fields)
        return cls(spaces=spaces.values(), fields=fields)

    def __repr__(self) -> str:
        return (
            f"DatasetSchema(spaces={list(self.spaces)}, fields={list(self.fields)})"
        )


def _infer_space(role: str, per_step: np.ndarray) -> SpaceSpec:
    """Build a candidate SpaceSpec for one field from a per-step example.

    Structural inference (shape/dtype/bounds/layout) is identical for all
    roles; the role only prefixes the generated key so that spaces never
    merge across roles — stacking observations with actions is meaningless
    even when their formats coincide.
    """
    shape = tuple(per_step.shape)
    dtype = per_step.dtype
    low = high = layout = None
    if dtype == np.uint8 and per_step.ndim >= 2:
        base = "image"
        low, high = 0, 255
        if per_step.ndim == 3:
            first, last = shape[0], shape[-1]
            if last in (1, 3, 4) and first not in (1, 3, 4):
                layout = "HWC"
            elif first in (1, 3, 4) and last not in (1, 3, 4):
                layout = "CHW"
    elif per_step.ndim == 1:
        base = "vector"
    elif per_step.ndim == 0:
        base = "scalar"
    else:
        base = "tensor"
    if role == "reward":
        base = "reward"
    elif role == "action":
        base = f"action_{base}"
    return SpaceSpec(key=base, shape=shape, dtype=dtype.name, low=low, high=high, layout=layout)


def _collapse_single_action_space(
    spaces: dict[str, SpaceSpec], fields: list[FieldSpec]
) -> None:
    """When all action fields share one structural space, call it plainly
    "action" — the structural suffix only earns its place when several
    distinct action formats must be told apart."""
    action_spaces = {f.space for f in fields if f.role == "action"}
    if len(action_spaces) != 1:
        return
    only = next(iter(action_spaces))
    if only == "action" or "action" in spaces:
        return
    spec = spaces.pop(only)
    spec.key = "action"
    spaces["action"] = spec
    for field in fields:
        if field.space == only:
            field.space = "action"


def _assign_space(spaces: dict[str, SpaceSpec], candidate: SpaceSpec) -> str:
    """Reuse a structurally identical space or register the candidate under a
    unique generated key."""
    for key, existing in spaces.items():
        if (
            existing.shape == candidate.shape
            and existing.dtype == candidate.dtype
            and key.rstrip("0123456789_") == candidate.key
        ):
            return key
    key = candidate.key
    suffix = 0
    while key in spaces:
        suffix += 1
        key = f"{candidate.key}_{suffix}"
    candidate.key = key
    spaces[key] = candidate
    return key