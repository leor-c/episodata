"""Optional interop with external frameworks.

Unlike the rest of ``episodata``, this module speaks a specific framework's
vocabulary. It stays out of the core import graph — ``import episodata``
never imports this module, and this module never imports a framework at top
level — so a missing optional dependency only surfaces when one of these
functions is actually called.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema import SEP, DatasetSchema, FieldSpec, _image_layout


def _require_gymnasium():
    try:
        import gymnasium.spaces as spaces
    except ImportError as e:
        raise ImportError(
            "schema_from_gym_spaces requires the optional 'gymnasium' "
            'package; install with pip install "episodata[gym]"'
        ) from e
    return spaces


def _uniform_bound(arr: np.ndarray) -> float | None:
    """A field bound is one scalar; a space's per-element bound array
    collapses to that scalar only if every element agrees."""
    arr = np.asarray(arr)
    first = arr.flat[0]
    return float(first) if np.all(arr == first) else None


def _leaf_field_spec(key: str, space: Any, role: str, spaces: Any) -> FieldSpec:
    if isinstance(space, spaces.Box):
        shape = tuple(int(s) for s in space.shape)
        dtype = np.dtype(space.dtype)
        return FieldSpec(
            key=key,
            shape=shape,
            dtype=dtype.name,
            role=role,
            low=_uniform_bound(space.low),
            high=_uniform_bound(space.high),
            layout=_image_layout(shape, dtype),
        )
    if isinstance(space, spaces.Discrete):
        start = int(getattr(space, "start", 0))
        return FieldSpec(
            key=key, shape=(), dtype="int64", role=role,
            low=start, high=start + int(space.n) - 1,
        )
    if isinstance(space, spaces.MultiDiscrete):
        nvec = np.asarray(space.nvec)
        start = np.asarray(getattr(space, "start", np.zeros_like(nvec)))
        return FieldSpec(
            key=key,
            shape=tuple(int(s) for s in nvec.shape),
            dtype=np.dtype(space.dtype).name,
            role=role,
            low=_uniform_bound(start),
            high=_uniform_bound(start + nvec - 1),
        )
    if isinstance(space, spaces.MultiBinary):
        n = space.n
        shape = (int(n),) if isinstance(n, (int, np.integer)) else tuple(int(s) for s in n)
        return FieldSpec(
            key=key, shape=shape, dtype=np.dtype(space.dtype).name, role=role,
            low=0, high=1,
        )
    raise NotImplementedError(
        f"schema_from_gym_spaces doesn't support {type(space).__name__} "
        f"(field {key!r}); build its FieldSpec manually and combine it with "
        f"the rest via DatasetSchema(fields=[...])"
    )


def _walk_space(key: str | None, space: Any, role: str, spaces: Any) -> list[FieldSpec]:
    """Flatten a space into FieldSpecs: Dict/Tuple recurse into per-member
    fields with SEP-joined path keys (mirrors normalize._flatten's handling
    of nested dict observations/actions); anything else is a leaf. A bare
    (non-container) space at the top level (``key is None``) takes the
    role's own name, exactly like normalize.add_group's default_key."""
    if isinstance(space, spaces.Dict):
        members = space.spaces.items()
    elif isinstance(space, spaces.Tuple):
        members = ((str(i), s) for i, s in enumerate(space.spaces))
    else:
        return [_leaf_field_spec(key if key is not None else role, space, role, spaces)]
    fields: list[FieldSpec] = []
    for name, sub in members:
        sub_key = f"{key}{SEP}{name}" if key is not None else name
        fields.extend(_walk_space(sub_key, sub, role, spaces))
    return fields


def schema_from_gym_spaces(observation_space: Any, action_space: Any) -> DatasetSchema:
    """Build a DatasetSchema directly from a Gymnasium env's spaces, no
    example data needed.

    Unlike ``DatasetSchema.infer`` (which reads shape/dtype off example
    data), this reads structure and bounds off the spaces themselves: ``Box``
    -> shape/dtype/low/high (plus image layout for uint8 arrays), ``Discrete``
    -> a scalar int64 bounded ``[start, start + n - 1]``,
    ``MultiDiscrete``/``MultiBinary`` -> integer vectors, and ``Dict``/``Tuple``
    recurse into per-member fields with flattened path keys (``SEP``) — the
    same flattening ``normalize.py`` performs for nested dict observations/
    actions. A scalar float32 ``reward`` field is always added, matching
    Gymnasium's ``step()`` contract.

    A ``Box``/``MultiDiscrete`` bound that isn't uniform across every element
    is dropped (``None``) — ``FieldSpec.low``/``high`` are single scalars,
    so per-dimension bounds can't be represented.

    Requires the optional ``gymnasium`` package
    (``pip install "episodata[gym]"``).
    """
    spaces = _require_gymnasium()
    fields = _walk_space(None, observation_space, "observation", spaces)
    fields += _walk_space(None, action_space, "action", spaces)
    fields.append(FieldSpec(key="reward", shape=(), dtype="float32", role="reward"))
    return DatasetSchema(fields=fields)
