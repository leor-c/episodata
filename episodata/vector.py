"""Online collection from N parallel (vectorized) environments.

A :class:`VectorWriter` drives one ongoing episode per environment and
handles their staggered boundaries: when environment ``i`` reports
``terminated``/``truncated``, its episode is finalized, and on the *next*
:meth:`step` call that environment's observation starts a fresh episode as
its reset row. This is next-step autoreset (the Gymnasium 1.0 vector
default) and matches the action-in layout exactly — the post-done step's
observation, ignored action and zero reward *are* episodata's row 0.

The writer is framework-agnostic: it takes plain arrays with a leading
``num_envs`` dimension and never imports an environment library. Vec envs
using the same-step convention (the final observation delivered inside
``infos`` on the done step) should drive per-episode writers directly —
interleaved open episodes are fully supported by ``Dataset.new_episode``.

::

    vec = dataset.vector_writer()
    obs, infos = envs.reset(seed=0)
    vec.reset(obs)
    while collecting:
        actions = policy(obs)
        obs, rewards, terminated, truncated, infos = envs.step(actions)
        vec.step(obs, actions=actions, rewards=rewards,
                 terminated=terminated, truncated=truncated)
    vec.close()
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .normalize import normalize_episode

if TYPE_CHECKING:
    from .dataset import Dataset


class VectorWriter:
    """Append handle for N parallel environments (next-step autoreset).

    Unlike the stateless :class:`EpisodeWriter`, this writer carries state
    (the per-env pending-reset mask), so it cannot be resumed across
    processes; finish collection with :meth:`close`. Individual episodes it
    leaves ongoing (``close(truncate=False)``) remain resumable by id via
    ``dataset.resume_episode``.
    """

    def __init__(self, dataset: "Dataset", num_envs: int | None = None):
        if num_envs is not None and num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self._dataset = dataset
        self._num_envs = num_envs
        self._episode_ids: list[int] = []
        self._needs_reset: np.ndarray | None = None
        self._closed = False

    @property
    def num_envs(self) -> int | None:
        return self._num_envs

    @property
    def episode_ids(self) -> tuple[int, ...]:
        """The most recent episode id per environment (for an env that is
        done and awaiting reset, the just-finalized episode)."""
        return tuple(self._episode_ids)

    def reset(
        self,
        observations: Mapping[str, Any] | Any,
        infos: Mapping[str, Any] | None = None,
    ) -> None:
        """Start one episode per environment from what ``envs.reset()``
        returned (arrays with a leading ``num_envs`` dimension).

        Resetting mid-collection is an external cutoff: still-ongoing
        episodes are finalized as ``truncated`` before the fresh episodes
        start.
        """
        self._check_open()
        fields, roles, n = self._normalize(observations=observations, infos=infos)
        if self._num_envs is None:
            self._num_envs = n
        elif n != self._num_envs:
            raise ValueError(f"reset data has {n} rows for {self._num_envs} envs")
        for episode_id in self._episode_ids:
            if self._dataset.backend.episode_ongoing(episode_id):
                self._dataset.end_episode(episode_id, truncated=True)
        self._episode_ids = [self._start_episode(fields, roles, i) for i in range(n)]
        self._needs_reset = np.zeros(n, dtype=bool)

    def step(
        self,
        observations: Mapping[str, Any] | Any,
        actions: Mapping[str, Any] | Any = None,
        rewards: Any = None,
        terminated: Any = None,
        truncated: Any = None,
        infos: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one ``envs.step()`` across all environments.

        All values carry a leading ``num_envs`` dimension; ``terminated``
        and ``truncated`` are boolean arrays (``None`` means all False). A
        True signal finalizes that environment's episode; on the next call
        its observation becomes the reset row of a fresh episode and its
        action/reward are ignored (next-step autoreset).
        """
        self._check_open()
        if self._needs_reset is None:
            raise ValueError("call reset() first")
        terminated = self._flags(terminated, "terminated")
        truncated = self._flags(truncated, "truncated")
        step: dict[str, Any] = {"observations": observations}
        if actions is not None:
            step["actions"] = actions
        if rewards is not None:
            step["rewards"] = rewards
        if infos is not None:
            step["infos"] = infos
        fields, roles, n = self._normalize(**step)
        if n != self._num_envs:
            raise ValueError(f"step data has {n} rows for {self._num_envs} envs")

        for i in np.flatnonzero(self._needs_reset):
            self._episode_ids[i] = self._start_episode(fields, roles, i)
        alive = np.flatnonzero(~self._needs_reset)
        if alive.size:
            self._dataset._add_step_batch(
                [self._episode_ids[i] for i in alive],
                {key: arr[alive] for key, arr in fields.items()},
            )
        self._needs_reset[:] = False
        for i in np.flatnonzero(terminated | truncated):
            self._dataset.end_episode(
                self._episode_ids[i], bool(terminated[i]), bool(truncated[i])
            )
            self._needs_reset[i] = True

    def close(self, truncate: bool = True) -> None:
        """Stop collecting and finalize still-ongoing episodes.

        Stopping collection is an external cutoff, so episodes are marked
        ``truncated`` by default (unlike ``EpisodeWriter.__exit__``, which
        ends its single episode with neither flag). ``truncate=False``
        leaves them ongoing, resumable via ``dataset.resume_episode``.
        """
        if self._closed:
            return
        if truncate:
            for episode_id in self._episode_ids:
                if self._dataset.backend.episode_ongoing(episode_id):
                    self._dataset.end_episode(episode_id, truncated=True)
        self._closed = True

    def _normalize(self, **groups: Any) -> tuple[dict[str, np.ndarray], dict[str, str], int]:
        """Normalize batched groups, treating ``num_envs`` as the leading dim.

        Termination flags never pass through here: normalization enforces
        the single-episode rule that a True flag may only close the final
        row, which a batch like ``terminated=[False, True]`` would violate.
        """
        normalized = normalize_episode(groups)
        return normalized.fields, normalized.roles, normalized.length

    def _start_episode(
        self, fields: Mapping[str, np.ndarray], roles: Mapping[str, str], i: int
    ) -> int:
        """Open a fresh episode for env ``i`` and write its reset row from
        the observation/info rows; other roles are ignored (row 0 zero-fills
        action and reward)."""
        observations = {k: fields[k][i] for k in fields if roles[k] == "observation"}
        infos = {k: fields[k][i] for k in fields if roles[k] == "info"}
        episode_id = self._dataset.backend.create_episode()
        self._dataset._write_reset_row(episode_id, observations, infos or None)
        return episode_id

    def _flags(self, value: Any, name: str) -> np.ndarray:
        if value is None:
            return np.zeros(self._num_envs, dtype=bool)
        arr = np.asarray(value, dtype=bool)
        if arr.shape != (self._num_envs,):
            raise ValueError(
                f"{name} must have shape ({self._num_envs},), got {arr.shape}"
            )
        return arr

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("writer is closed")

    def __enter__(self) -> "VectorWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
