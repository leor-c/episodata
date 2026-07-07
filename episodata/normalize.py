"""Normalization of user-provided episode data into flat temporal fields.

The canonical episode dict accepted from users looks like::

    {
        "observations": {"front_camera": array[T, ...], "state": array[T, ...]},
        "actions": {"action": array[T, ...]},   # or a bare array
        "rewards": array[T],                     # optional
        "terminated": True,                      # bool or per-step array
        "truncated": False,
        "infos": {"success": array[T]},          # optional
    }

Internally everything becomes a flat mapping of field key -> array with a
leading time dimension, plus episode-level terminated/truncated flags.
Nested dicts (complex action/observation structures, e.g. Minecraft-style
``{"keyboard": {"w": ...}}``) are flattened into stable path keys such as
``"keyboard/w"``; the hierarchy is reconstructed from the schema, not from
nesting in storage.

Temporal alignment is "action-in": row ``t`` holds the action and reward
that *led to* observation ``t``. Row 0 is the reset row — the initial
observation with dummy zero action/reward (see ``Dataset.add_reset``).
Data recorded in the "action-out" convention (action taken *at* the row's
observation) is converted with :func:`shift_action_out`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

#: Separator for flattening nested field structures into stable path keys.
SEP = "/"

_OBS_KEYS = ("observations", "observation", "obs")
_ACTION_KEYS = ("actions", "action")
_REWARD_KEYS = ("rewards", "reward")
_TERMINATED_KEYS = ("terminated", "terminals", "terminations", "dones")
_TRUNCATED_KEYS = ("truncated", "truncations")
_INFO_KEYS = ("infos", "info")

_ALL_TOP_LEVEL = (
    _OBS_KEYS + _ACTION_KEYS + _REWARD_KEYS + _TERMINATED_KEYS + _TRUNCATED_KEYS + _INFO_KEYS
)


@dataclasses.dataclass
class NormalizedEpisode:
    """Flat per-field arrays (leading time dim) plus episode-level flags."""

    fields: dict[str, np.ndarray]
    roles: dict[str, str]
    length: int
    terminated: bool
    truncated: bool


def normalize_episode(episode: Mapping[str, Any]) -> NormalizedEpisode:
    """Convert a canonical episode dict into a :class:`NormalizedEpisode`.

    All temporal fields must share the same length ``T``.
    """
    unknown = [k for k in episode if k not in _ALL_TOP_LEVEL]
    if unknown:
        raise ValueError(
            f"unknown episode keys {unknown}; expected observations/actions/"
            f"rewards/terminated/truncated/infos"
        )

    fields: dict[str, np.ndarray] = {}
    roles: dict[str, str] = {}

    def add(key: str, value: Any, role: str) -> None:
        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"field group {key!r} is empty")
            for sub_key, sub_value in value.items():
                add(f"{key}{SEP}{sub_key}", sub_value, role)
            return
        if key in fields:
            raise ValueError(f"duplicate field key {key!r} across groups")
        arr = np.asarray(value)
        if arr.ndim == 0:
            raise ValueError(f"field {key!r} must have a leading time dimension")
        fields[key] = arr
        roles[key] = role

    def add_group(group: Any, role: str, default_key: str) -> None:
        if group is None:
            return
        if isinstance(group, Mapping):
            for key, value in group.items():
                add(key, value, role)
        else:
            add(default_key, group, role)

    add_group(_first(episode, _OBS_KEYS), "observation", "observation")
    add_group(_first(episode, _ACTION_KEYS), "action", "action")
    reward = _first(episode, _REWARD_KEYS)
    if reward is not None:
        add("reward", reward, "reward")
    add_group(_first(episode, _INFO_KEYS), "info", "info")

    if not fields:
        raise ValueError("episode contains no temporal fields")

    lengths = {key: len(arr) for key, arr in fields.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"all temporal fields must share one length, got {lengths}")
    length = next(iter(lengths.values()))

    terminated = _first(episode, _TERMINATED_KEYS)
    truncated = _first(episode, _TRUNCATED_KEYS)
    for name, raw in (("terminated", terminated), ("truncated", truncated)):
        _validate_flag(name, raw, length)

    return NormalizedEpisode(
        fields=fields,
        roles=roles,
        length=length,
        terminated=_flag(terminated),
        truncated=_flag(truncated),
    )


def normalize_step(step: Mapping[str, Any]) -> NormalizedEpisode:
    """Normalize a single step (no time dimension) into a length-1 episode."""
    expanded: dict[str, Any] = {}
    for key, value in step.items():
        if key in _TERMINATED_KEYS or key in _TRUNCATED_KEYS:
            expanded[key] = value
        else:
            expanded[key] = _expand_step(value)
    return normalize_episode(expanded)


def _expand_step(value: Any) -> Any:
    """Recursively add a length-1 time dimension to the leaves of a step."""
    if isinstance(value, Mapping):
        return {k: _expand_step(v) for k, v in value.items()}
    return np.asarray(value)[None]


def shift_action_out(episode: NormalizedEpisode) -> NormalizedEpisode:
    """Convert an action-out episode to the canonical action-in alignment.

    Action/reward rows move one step later, so that row ``t`` holds what led
    to observation ``t``; row 0 gets zeros (the reset row). The source's
    final action/reward are dropped — the observation they produced was
    never recorded, so no transition could ever use them.
    """
    fields: dict[str, np.ndarray] = {}
    for key, arr in episode.fields.items():
        if episode.roles[key] in ("action", "reward"):
            shifted = np.zeros_like(arr)
            shifted[1:] = arr[:-1]
            fields[key] = shifted
        else:
            fields[key] = arr
    return dataclasses.replace(episode, fields=fields)


def _first(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    present = [k for k in keys if k in mapping]
    if len(present) > 1:
        raise ValueError(f"episode contains aliased keys {present}; use one")
    return mapping[present[0]] if present else None


def _flag(value: Any) -> bool:
    """Reduce a bool or per-step array of flags to an episode-level flag."""
    if value is None:
        return False
    arr = np.asarray(value)
    return bool(arr.any())


def _validate_flag(name: str, value: Any, length: int) -> None:
    """Per-step termination signals must match the episode length and, as in
    Gymnasium, may only be True on the final step — the episode ends there."""
    if value is None:
        return
    arr = np.asarray(value)
    if arr.ndim == 0:
        return
    if len(arr) != length:
        raise ValueError(
            f"{name} has length {len(arr)}, expected the episode length {length}"
        )
    if arr[:-1].any():
        raise ValueError(f"{name} may only be True on the final step")