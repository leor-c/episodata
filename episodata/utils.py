"""Optional interop with external frameworks.

Unlike the rest of ``episodata``, this module speaks a specific framework's
vocabulary. It stays out of the core import graph — ``import episodata``
never imports this module, and this module never imports a framework at top
level — so a missing optional dependency only surfaces when one of these
functions is actually called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .fields import Fields
from .schema import SEP, DatasetSchema, FieldSpec, _image_layout

if TYPE_CHECKING:
    from .segment import Segment


def _require_gymnasium():
    try:
        import gymnasium.spaces as spaces
    except ImportError as e:
        raise ImportError(
            "schema_from_gym_spaces requires the optional 'gymnasium' "
            'package; install with pip install "episodata[gym]"'
        ) from e
    return spaces


def _require_tensordict():
    try:
        import torch
        from tensordict import TensorDict
    except ImportError as e:
        raise ImportError(
            "batch_to_tensordict requires the optional 'torch' and "
            "'tensordict' packages; install with pip install torch tensordict"
        ) from e
    return torch, TensorDict


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


def _unflatten_role(fields: Fields) -> dict[str, Any]:
    """Turn a grouped role's SEP-joined flat keys into the nested dict
    TensorDict expects, mirroring how FieldGroup builds hierarchy from those
    same keys on attribute access (e.g. ``"keyboard/w"`` ->
    ``{"keyboard": {"w": ...}}``)."""
    nested: dict[str, Any] = {}
    for key, value in fields.items():
        *groups, leaf = key.split(SEP)
        node = nested
        for group in groups:
            node = node.setdefault(group, {})
        node[leaf] = value
    return nested


def _role_entry(value: Any) -> Any:
    """A bare-array role passes through; a grouped (``Fields``) role becomes
    a nested dict; an empty role (unused in this schema) is dropped."""
    if isinstance(value, Fields):
        return _unflatten_role(value) if len(value) else None
    return value


def _pad_time(tensor: Any, time_axis: int, pad: str) -> Any:
    """Grow ``tensor`` by one zero row along ``time_axis``, at the front
    (``pad="before"``) or back (``pad="after"``)."""
    import torch

    pad_shape = list(tensor.shape)
    pad_shape[time_axis] = 1
    zero_row = tensor.new_zeros(pad_shape)
    pieces = (zero_row, tensor) if pad == "before" else (tensor, zero_row)
    return torch.cat(pieces, dim=time_axis)


def _pad_entry(entry: Any, time_axis: int, pad: str) -> Any:
    if isinstance(entry, dict):
        return {k: _pad_entry(v, time_axis, pad) for k, v in entry.items()}
    return _pad_time(entry, time_axis, pad)


def batch_to_tensordict(
    batch: "Segment",
    device: Any = None,
    include_all_observations: bool = False,
    pad: Literal["before", "after"] | None = None,
) -> Any:
    """Convert a :class:`~episodata.segment.Segment` or
    :class:`~episodata.segment.Batch` to a :class:`tensordict.TensorDict`,
    one entry per role present in the schema (``observation``, ``action``,
    ``reward``, ``next_observation``, ``info``/``next_info`` when non-empty,
    and ``terminated``/``truncated``/``mask`` when set). A grouped role
    (nested dict observation/action) becomes a nested dict entry, mirroring
    its ``/``-separated field keys.

    ``device``, if given, is applied inside the same
    :meth:`~episodata.segment.Segment.map` pass that converts arrays to
    tensors — *before* ``observation``/``next_observation`` are sliced out
    of it — so both stay views of one on-device tensor per field and the
    conversion never duplicates their overlapping rows. Calling ``.to(device)``
    on the returned TensorDict instead would move each role's tensor
    independently and copy that shared storage; pass ``device`` here, not on
    the result.

    ``include_all_observations``, if set, adds ``all_observations`` (and
    ``all_infos`` when the info role is used) — the ``L + 1``-row buffer
    ``observation``/``next_observation`` are already zero-copy slices of, so
    this adds no data, just an entry spanning both. Because that entry is one
    row longer than every other, and a ``TensorDict``'s ``batch_size`` must
    equal every entry's leading dims exactly, ``batch_size`` then drops the
    time dim and keeps only what every entry actually shares (the batch dim,
    for a :class:`~episodata.segment.Batch`; nothing, for a lone
    :class:`~episodata.segment.Segment`) — so e.g. ``td["mask"]`` stays a
    ``[B, L]`` tensor even though ``batch_size`` is only ``(B,)``.

    ``pad``, if set, makes every entry ``L + 1`` rows long so ``batch_size``
    covers the full time dim (``[B, L + 1]``, or ``[L + 1]`` unbatched) with
    no shrinking. ``observation``/``next_observation``/``info``/``next_info``
    are dropped in favor of ``all_observations``/``all_infos`` (real data,
    already ``L + 1`` long — this is what ``include_all_observations`` adds,
    so the two options conflict and can't be combined). ``action``,
    ``reward``, and ``terminated``/``truncated``/``mask`` have no such
    ``L + 1`` counterpart, so they're grown by one zero row instead — that
    row is a placeholder, not real data:

    - ``pad="before"`` prepends the zero row, so entry ``t`` reads as
      *"the action/reward/flag that led into ``all_observations[t]``"*
      (undefined, i.e. zero, at ``t=0`` — the window's first observation).
    - ``pad="after"`` appends the zero row, so entry ``t`` reads as
      *"the action/reward/flag taken at ``all_observations[t]``"* (undefined
      at ``t=L`` — the window's last observation has no known outgoing
      action).

    Pick whichever side matches how the rest of your pipeline aligns
    actions to observations. Because this introduces placeholder data, it's
    opt-in only: the default (``pad=None``) never pads.

    Requires the optional ``torch`` and ``tensordict`` packages.
    """
    torch, TensorDict = _require_tensordict()

    if pad is not None:
        if pad not in ("before", "after"):
            raise ValueError(f"pad must be 'before', 'after', or None, got {pad!r}")
        if include_all_observations:
            raise ValueError(
                "pad already includes all_observations/all_infos in place of "
                "observation/next_observation/info/next_info; pass "
                "include_all_observations=False (the default) with pad"
            )

    def convert(arr):
        tensor = torch.as_tensor(arr)
        return tensor if device is None else tensor.to(device)

    converted = batch.map(convert)

    if pad is not None and converted._squeeze:
        raise ValueError("pad requires a Segment/Batch with a time dimension")

    data: dict[str, Any] = {}
    roles = (
        ("action", "reward")
        if pad is not None
        else ("observation", "action", "reward", "next_observation", "info", "next_info")
    )
    for role in roles:
        entry = _role_entry(getattr(converted, role))
        if entry is not None:
            if pad is not None:
                entry = _pad_entry(entry, converted._time_axis, pad)
            data[role] = entry

    if include_all_observations or pad is not None:
        data["all_observations"] = _role_entry(converted.all_observations)
        all_infos_entry = _role_entry(converted.all_infos)
        if all_infos_entry is not None:
            data["all_infos"] = all_infos_entry

    batch_size = None
    for flag in ("mask", "terminated", "truncated"):
        value = getattr(converted, flag)
        if value is not None:
            if pad is not None:
                value = _pad_time(value, converted._time_axis, pad)
            data[flag] = value
            if batch_size is None:
                # Flag arrays carry no per-field trailing dims, so their
                # shape is exactly the batch dims. Drop the time dim (its
                # last axis) when all_observations joined the entries with
                # one extra row and wasn't itself padded to match — unless
                # it was already squeezed away, in which case there's no
                # time dim left in the flags to drop.
                if pad is not None or not include_all_observations or converted._squeeze:
                    batch_dims = value.ndim
                else:
                    batch_dims = converted._time_axis
                batch_size = value.shape[:batch_dims]
    if batch_size is None:
        raise ValueError(
            "batch_to_tensordict needs terminated/truncated/mask to infer "
            "batch_size; construct the Segment/Batch with at least one set"
        )
    return TensorDict(data, batch_size=batch_size)
