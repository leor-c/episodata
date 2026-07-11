"""Normalization of user-provided episode data into flat temporal fields.

Whole-episode bulk import (:func:`normalize_full_episode`, used by
``Dataset.from_episodes``/``add_episode`` under the default
``alignment="action_in"``) speaks env steps: every temporal field carries one
entry per step, and the reset observation is a separate, explicitly named
``initial_observation`` — Gymnasium's vocabulary, ``env.reset()``'s return::

    {
        "initial_observation": {"front_camera": array[...], "state": array[...]},
        "observations": {"front_camera": array[T, ...], "state": array[T, ...]},
        "actions": {"action": array[T, ...]},   # or a bare array
        "rewards": array[T],                     # optional
        "terminated": True,                      # bool or per-step array[T]
        "truncated": False,
        "infos": {"success": array[T]},           # optional, needs initial_info
        "initial_info": {"success": array_like},  # optional, pairs with infos
    }

The reset row's dummy zero action/reward — mirroring what
``Dataset.new_episode`` writes online — is synthesized here, never supplied
by the caller. ``infos`` pairing is all-or-nothing: an arbitrary info dict
has no universal zero sentinel, so it is never zero-filled — if ``infos`` is
supplied, ``initial_info`` must cover the same fields (Gymnasium's ``info``
accompanies ``reset()`` too), and vice versa; both may be omitted entirely.

Data recorded in the "action-out" convention (action taken *at* the row's
observation, D4RL-style) is equal-length ``T`` as well and converted at this
boundary (:func:`normalize_action_out_episode`): the optional
``final_observation`` / ``final_info`` keys carry the observation produced by
the final action, matching ``ActionOutWriter.end``'s kwargs; without them the
final action/reward are dropped (their resulting observation was never
recorded, so no transition could use them).

Continuing an already-open episode (``add_steps``, and single-step
``add_step``/``normalize_step``) appends complete storage rows — every field
there, including observations, actions, rewards and infos, shares one equal
length (:func:`normalize_episode`).

Internally everything becomes a flat mapping of field key -> array with a
leading time dimension, plus episode-level terminated/truncated flags.
Every role follows one rule: a dict source spreads its member keys, a bare
source gets the role's canonical name (``observation``/``action``/
``reward``/``info``). Nested dicts (complex structures, e.g. Minecraft-style
``{"keyboard": {"w": ...}}``) are flattened into stable path keys such as
``"keyboard/w"``; the hierarchy is reconstructed from the schema, not from
nesting in storage.

Temporal alignment is "action-in": row ``t`` holds the action and reward
that *led to* observation ``t``. Row 0 is the reset row.
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

# Boundary-row keys for whole-episode bulk import. Deliberately not part of
# _ALL_TOP_LEVEL/_flatten: they are only meaningful to normalize_full_episode
# and normalize_action_out_episode, never to add_steps/normalize_step.
_INITIAL_OBS_KEY = "initial_observation"
_INITIAL_INFO_KEY = "initial_info"
_FINAL_OBS_KEY = "final_observation"
_FINAL_INFO_KEY = "final_info"


@dataclasses.dataclass
class NormalizedEpisode:
    """Flat per-field arrays (leading time dim) plus episode-level flags."""

    fields: dict[str, np.ndarray]
    roles: dict[str, str]
    length: int
    terminated: bool
    truncated: bool


def _flatten(episode: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Flatten a canonical episode dict into flat field arrays plus roles."""
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
    add_group(_first(episode, _REWARD_KEYS), "reward", "reward")
    add_group(_first(episode, _INFO_KEYS), "info", "info")

    if not fields:
        raise ValueError("episode contains no temporal fields")

    return fields, roles


def normalize_episode(episode: Mapping[str, Any]) -> NormalizedEpisode:
    """Convert a canonical episode dict into a :class:`NormalizedEpisode`.

    All temporal fields must share the same length ``T``.
    """
    fields, roles = _flatten(episode)

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


def resolve_alignment(episode: Mapping[str, Any], alignment: str | None = None) -> str:
    """Determine a bulk-import episode dict's alignment from its boundary keys.

    The dict is self-describing: ``initial_observation``/``initial_info``
    mark action-in, ``final_observation``/``final_info`` mark action-out,
    and carrying both is a contradiction. An explicit ``alignment`` is
    validated against the keys; it is only *required* for action-out data
    without its final observation, which carries no boundary key.
    """
    initial = [k for k in (_INITIAL_OBS_KEY, _INITIAL_INFO_KEY) if k in episode]
    final = [k for k in (_FINAL_OBS_KEY, _FINAL_INFO_KEY) if k in episode]
    if initial and final:
        raise ValueError(
            f"{initial[0]!r} (action-in) and {final[0]!r} (action-out) are mutually exclusive"
        )
    inferred = "action_in" if initial else "action_out" if final else None
    if alignment is None:
        if inferred is None:
            raise ValueError(
                f"cannot determine alignment: pass {_INITIAL_OBS_KEY!r} (action-in) or "
                f"{_FINAL_OBS_KEY!r} (action-out), or alignment='action_out' for "
                f"action-out data without its final observation"
            )
        return inferred
    if alignment not in ("action_in", "action_out"):
        raise ValueError(f"unknown alignment {alignment!r}")
    if inferred is not None and inferred != alignment:
        raise ValueError(
            f"episode carries {(initial or final)[0]!r} but alignment={alignment!r} was passed"
        )
    return alignment


def normalize_full_episode(episode: Mapping[str, Any]) -> NormalizedEpisode:
    """Convert a canonical whole-episode dict (bulk import) into a :class:`NormalizedEpisode`.

    ``initial_observation`` (required) is the reset observation; every
    temporal field carries one entry per env step. The reset row is
    assembled here — real observation/info data, dummy zero action/reward —
    mirroring ``Dataset.new_episode``.
    """
    if _INITIAL_OBS_KEY not in episode:
        raise ValueError(
            f"action-in bulk import requires {_INITIAL_OBS_KEY!r} (the reset observation)"
        )
    reset_step: dict[str, Any] = {"observations": episode[_INITIAL_OBS_KEY]}
    if _INITIAL_INFO_KEY in episode:
        reset_step["infos"] = episode[_INITIAL_INFO_KEY]
    reset = normalize_step(reset_step)

    core_episode = {
        k: v for k, v in episode.items() if k not in (_INITIAL_OBS_KEY, _INITIAL_INFO_KEY)
    }
    core = normalize_episode(core_episode)

    _validate_boundary_row(reset, core, _INITIAL_OBS_KEY, _INITIAL_INFO_KEY)

    fields: dict[str, np.ndarray] = {}
    for key, arr in core.fields.items():
        if key in reset.fields:
            fields[key] = np.concatenate([reset.fields[key], arr])
        else:  # action/reward: no reset-row counterpart, zero-fill row 0
            padded = np.zeros((len(arr) + 1, *arr.shape[1:]), dtype=arr.dtype)
            padded[1:] = arr
            fields[key] = padded

    return NormalizedEpisode(
        fields=fields,
        roles=core.roles,
        length=core.length + 1,
        terminated=core.terminated,
        truncated=core.truncated,
    )


def normalize_action_out_episode(episode: Mapping[str, Any]) -> NormalizedEpisode:
    """Convert an action-out whole-episode dict into a :class:`NormalizedEpisode`.

    Row ``t`` of the input holds the action taken *at* observation ``t``
    (D4RL-style). ``final_observation`` (optional) is the observation the
    final action produced: with it, nothing is dropped; without it, the
    final action/reward are dropped (their resulting observation was never
    recorded, so no transition could use them).
    """
    core_episode = {
        k: v for k, v in episode.items() if k not in (_FINAL_OBS_KEY, _FINAL_INFO_KEY)
    }
    core = normalize_episode(core_episode)

    if _FINAL_OBS_KEY not in episode:
        if _FINAL_INFO_KEY in episode:
            raise ValueError(f"{_FINAL_INFO_KEY!r} requires {_FINAL_OBS_KEY!r}")
        return shift_action_out(core)

    final_step: dict[str, Any] = {"observations": episode[_FINAL_OBS_KEY]}
    if _FINAL_INFO_KEY in episode:
        final_step["infos"] = episode[_FINAL_INFO_KEY]
    final_row = normalize_step(final_step)

    _validate_boundary_row(final_row, core, _FINAL_OBS_KEY, _FINAL_INFO_KEY)

    fields: dict[str, np.ndarray] = {}
    for key, arr in core.fields.items():
        if key in final_row.fields:
            fields[key] = np.concatenate([arr, final_row.fields[key]])
        else:  # action/reward: shift one row later; nothing dropped this time
            shifted = np.zeros((len(arr) + 1, *arr.shape[1:]), dtype=arr.dtype)
            shifted[1:] = arr
            fields[key] = shifted

    return NormalizedEpisode(
        fields=fields,
        roles=core.roles,
        length=core.length + 1,
        terminated=core.terminated,
        truncated=core.truncated,
    )


def _validate_boundary_row(
    row: NormalizedEpisode, core: NormalizedEpisode, obs_key: str, info_key: str
) -> None:
    """A boundary row must cover exactly the core's observation/info fields —
    this is what makes the infos pairing all-or-nothing."""
    row_keys = {k for k in core.fields if core.roles[k] in ("observation", "info")}
    if set(row.fields) != row_keys:
        raise ValueError(
            f"{obs_key}/{info_key} must cover exactly the same fields as "
            f"observations/infos; got {sorted(row.fields)} vs {sorted(row_keys)}"
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