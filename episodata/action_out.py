"""Adapter for the action-out convention.

episodata stores episodes action-in: row ``t`` holds the action and reward
that *led to* observation ``t``, with a dummy reset row at index 0. Many
classic RL pipelines instead produce action-out steps — ``(o_t, a_t, r_t)``
where ``a_t`` is the action taken *at* ``o_t``.

Action-out users need adapters only at the write boundaries; every read is
alignment-agnostic — segments and transitions alike name the pairing
explicitly (``observations``/``actions``/``next_observations``).

- bulk import: ``Dataset.from_episodes(..., alignment="action_out")``,
  optionally with ``final_observation`` / ``final_info`` keys (mirroring
  :meth:`ActionOutWriter.end`) to keep the final transition
- online collection: :class:`ActionOutWriter`, wrapping a normal writer
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .episode import Episode, EpisodeWriter
from .normalize import (
    _ACTION_KEYS,
    _INFO_KEYS,
    _OBS_KEYS,
    _REWARD_KEYS,
    _TERMINATED_KEYS,
    _TRUNCATED_KEYS,
    _first,
    _flag,
)


class ActionOutWriter:
    """Streams action-out steps into canonical action-in storage.

    Each :meth:`add_step` takes ``(o_t, a_t, r_t)`` with the action taken
    *at* that observation. The observation (and info) are written
    immediately; the action/reward are held back one step and written
    alongside the observation they produce.

    ::

        writer = ActionOutWriter(dataset.new_episode())
        writer.add_step({"observations": obs, "actions": a, "rewards": r})
        ...
        writer.end(terminated=True, final_observation=last_obs)
    """

    def __init__(self, writer: EpisodeWriter):
        self._writer = writer
        self._started = False
        self._pending_action: Any = None
        self._pending_reward: Any = None

    @property
    def episode_id(self) -> int:
        return self._writer.episode_id

    def add_step(self, step: Mapping[str, Any]) -> None:
        """Append one action-out step ``(o_t, a_t, r_t[, info_t])``.

        The step may carry ``terminated`` / ``truncated`` describing ``o_t``
        itself; a True signal finalizes the episode immediately (any action
        in such a step is dropped — nothing can follow a terminal state).
        """
        observations = _first(step, _OBS_KEYS)
        infos = _first(step, _INFO_KEYS)
        if observations is None:
            raise ValueError("an action-out step must contain observations")
        if self._started:
            self._writer.add_step(self._row(observations, infos))
        else:
            self._writer._write_reset(observations, infos=infos)
            self._started = True
        self._pending_action = _first(step, _ACTION_KEYS)
        self._pending_reward = _first(step, _REWARD_KEYS)
        terminated = _flag(_first(step, _TERMINATED_KEYS))
        truncated = _flag(_first(step, _TRUNCATED_KEYS))
        if terminated or truncated:
            self._writer.end(terminated=terminated, truncated=truncated)

    def end(
        self,
        terminated: bool = False,
        truncated: bool = False,
        final_observation: Mapping[str, Any] | Any = None,
        final_info: Mapping[str, Any] | None = None,
    ) -> Episode:
        """Finalize the episode.

        Pass ``final_observation`` (what the last ``env.step`` returned) to
        keep the final transition. Without it, the held action/reward are
        dropped — the observation they produced was never recorded.
        """
        if final_observation is not None:
            self._writer.add_step(self._row(final_observation, final_info))
        return self._writer.end(terminated=terminated, truncated=truncated)

    def _row(self, observations: Any, infos: Any) -> dict[str, Any]:
        row: dict[str, Any] = {"observations": observations}
        if infos is not None:
            row["infos"] = infos
        if self._pending_action is not None:
            row["actions"] = self._pending_action
        if self._pending_reward is not None:
            row["rewards"] = self._pending_reward
        return row
