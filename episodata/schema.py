"""Logical schema: fields and the dataset schema.

The schema defines what the data *means*. It is independent of how any
backend physically stores the data. Once a dataset is created, the persisted
schema is authoritative; it is never re-inferred on reopen.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

SCHEMA_VERSION = 2

#: Separator for flattening nested field structures into stable path keys.
SEP = "/"

#: Field roles. Observations, actions and rewards are temporal fields that
#: exist at every step of an episode. "info" covers auxiliary per-step data.
ROLES = ("observation", "action", "reward", "info")


@dataclasses.dataclass
class FieldSpec:
    """A named field with a stable identity and a declared per-step format.

    The field key is the stable logical identifier used across the storage
    boundary. The format (shape/dtype, optional bounds and layout) is what
    backends allocate against and writes are validated against — the same
    per-leaf model as a Gymnasium ``Dict`` space, where every key carries
    its own ``Box``.
    """

    key: str
    shape: tuple[int, ...]
    dtype: str
    role: str = "observation"
    low: float | None = None
    high: float | None = None
    layout: str | None = None
    semantic_type: str | None = None
    optional: bool = False
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shape = tuple(int(s) for s in self.shape)
        self.dtype = np.dtype(self.dtype).name
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}, expected one of {ROLES}")

    def matches(self, per_step_array: np.ndarray) -> bool:
        """Whether a per-step value structurally belongs to this field."""
        arr = np.asarray(per_step_array)
        return tuple(arr.shape) == self.shape and arr.dtype == np.dtype(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FieldSpec:
        return cls(
            key=d["key"],
            shape=tuple(d["shape"]),
            dtype=d["dtype"],
            role=d.get("role", "observation"),
            low=d.get("low"),
            high=d.get("high"),
            layout=d.get("layout"),
            semantic_type=d.get("semantic_type"),
            optional=d.get("optional", False),
            metadata=dict(d.get("metadata", {})),
        )


class DatasetSchema:
    """The persistent logical schema of a dataset."""

    def __init__(self, fields: Iterable[FieldSpec]):
        self.fields: dict[str, FieldSpec] = {}
        for field in fields:
            if field.key in self.fields:
                raise ValueError(f"duplicate field key {field.key!r}")
            self.fields[field.key] = field

    # -- lookup ----------------------------------------------------------

    def field(self, key: str) -> FieldSpec:
        return self.fields[key]

    def field_keys(self, role: str | None = None) -> list[str]:
        if role is None:
            return list(self.fields)
        return [k for k, f in self.fields.items() if f.role == role]

    def resolve_fields(self, fields: list[str] | None) -> list[str]:
        """Resolve requested names to schema field keys.

        A name that is a path prefix (e.g. ``"keyboard"``) selects every
        field beneath it.
        """
        if fields is None:
            return self.field_keys()
        resolved: list[str] = []
        for name in fields:
            if name in self.fields:
                resolved.append(name)
                continue
            head = f"{name}{SEP}"
            members = [k for k in self.fields if k.startswith(head)]
            if not members:
                raise KeyError(
                    f"unknown field {name!r}; schema has {list(self.fields)}"
                )
            resolved.extend(members)
        return resolved

    # -- refinement (hybrid mode) -----------------------------------------

    def rename_field(self, old: str, new: str) -> None:
        if new in self.fields:
            raise ValueError(f"field key {new!r} already exists")
        field = self.fields.pop(old)
        field.key = new
        self.fields[new] = field

    # -- validation --------------------------------------------------------

    def validate_fields(self, fields: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Validate temporal field data against the schema and cast to
        logical dtype.

        Every array must carry a leading time dimension and match its
        field's declared per-step shape; all required (non-optional) fields
        must be present.
        """
        out: dict[str, np.ndarray] = {}
        for key, arr in fields.items():
            if key not in self.fields:
                raise KeyError(f"field {key!r} is not in the schema")
            spec = self.fields[key]
            arr = np.asarray(arr)
            expected = (len(arr), *spec.shape)
            if tuple(arr.shape) != expected:
                raise ValueError(
                    f"field {key!r}: got shape {tuple(arr.shape)}, expected {expected}"
                )
            out[key] = arr.astype(spec.dtype, copy=False)
        missing = [
            k for k, f in self.fields.items() if k not in out and not f.optional
        ]
        if missing:
            raise ValueError(f"missing required fields {missing}")
        return out

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "fields": [f.to_dict() for f in self.fields.values()],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DatasetSchema:
        version = d.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema version {version}; "
                f"this build reads version {SCHEMA_VERSION}"
            )
        return cls(fields=[FieldSpec.from_dict(f) for f in d["fields"]])

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

        Shape and dtype are inferred reliably; image bounds and layout come
        from structural heuristics. The episode's alignment is read from its
        boundary keys (``initial_observation`` marks action-in,
        ``final_observation`` action-out); ``alignment`` is only needed for
        action-out data without its final observation (see
        ``resolve_alignment``).
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
        fields = [
            _infer_field(key, normalized.roles[key], array[0])
            for key, array in normalized.fields.items()
        ]
        return cls(fields=fields)

    def __repr__(self) -> str:
        return f"DatasetSchema(fields={list(self.fields)})"


def _infer_field(key: str, role: str, per_step: np.ndarray) -> FieldSpec:
    """Build a FieldSpec for one field from a per-step example.

    Shape and dtype are read off the example. uint8 arrays with at least two
    dims are treated as images: bounds default to [0, 255] and a 3-dim shape
    with an unambiguous channel dim records its layout (HWC or CHW).
    """
    shape = tuple(per_step.shape)
    dtype = per_step.dtype
    low = high = layout = None
    if dtype == np.uint8 and per_step.ndim >= 2:
        low, high = 0, 255
        if per_step.ndim == 3:
            first, last = shape[0], shape[-1]
            if last in (1, 3, 4) and first not in (1, 3, 4):
                layout = "HWC"
            elif first in (1, 3, 4) and last not in (1, 3, 4):
                layout = "CHW"
    return FieldSpec(
        key=key,
        shape=shape,
        dtype=dtype.name,
        role=role,
        low=low,
        high=high,
        layout=layout,
        optional=(role == "info"),
    )
