# collector.py
from __future__ import annotations

from typing import Callable, List, Optional

import torch
from tensordict import TensorDict

from .episode import Episode
from .dataset import EpisodeDataset


class Collector:
    """
    VecEnv-friendly collector that keeps *active* episodes inside the dataset.

    Behavior:
      - On init, creates one active Episode per env and adds it to the dataset.
      - digest(batch_td): routes per-env rows; Episode auto-finalizes on done.
        When an episode completes, a new one is started immediately for that env
        and added to the dataset.

    Assumptions (next-step autoreset style):
      - The VERY FIRST digest call is a batch of first steps (each row has only
        'obs' or 'next_obs'). These are appended as partial first steps.
      - Subsequent digests carry full steps:
          {'action', 'next_obs', 'reward', 'done' OR ('terminated','truncated')}
    """

    def __init__(
        self,
        dataset: EpisodeDataset,
        num_envs: int,
        *,
        done_key: Optional[str] = "done",
        terminated_key: Optional[str] = "terminated",
        truncated_key: Optional[str] = "truncated",
        episode_metadata_factory: Optional[Callable[[int, int], dict]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.dataset = dataset
        self.num_envs = int(num_envs)
        self._done_key = done_key
        self._terminated_key = terminated_key
        self._truncated_key = truncated_key
        self._episode_metadata_factory = episode_metadata_factory
        self.device = device

        self._active: List[Episode] = []
        self._expect_first: List[bool] = [True] * self.num_envs

        for env_idx in range(self.num_envs):
            ep = self._new_episode(env_idx)
            self.dataset.add(ep)   # include active episodes in dataset
            self._active.append(ep)

    def active_episode(self, env_idx: int) -> Episode:
        return self._active[env_idx]

    def active_ids(self) -> List[int]:
        return [ep.episode_id for ep in self._active]

    def active_map(self) -> dict[int, int]:
        return {i: ep.episode_id for i, ep in enumerate(self._active)}

    @torch.no_grad()
    def digest(self, batch_td: TensorDict) -> List[int]:
        """
        Ingest a vectorized step batch (leading dim = num_envs).
        Returns list of episode_ids that completed in this call.
        """
        if not isinstance(batch_td, TensorDict):
            raise TypeError("digest(batch_td): expected a TensorDict.")
        bs = batch_td.batch_size
        if len(bs) == 0 or bs[0] != self.num_envs:
            raise ValueError(f"Expected batch_td.batch_size[0] == num_envs ({self.num_envs}), got {tuple(bs)}")

        completed: List[int] = []

        for env_idx in range(self.num_envs):
            row: TensorDict = batch_td[env_idx]
            ep = self._active[env_idx]

            if self._expect_first[env_idx]:
                # first-step obs only
                if "obs" not in row.keys() and "next_obs" not in row.keys():
                    raise KeyError(f"Expected first-step obs for env {env_idx}, got keys {list(row.keys())}")
                ep.extend(row)  # partial first step (Episode handles it)
                self._expect_first[env_idx] = False
                continue

            # full step
            full_step = self._to_full_step(row, device=self.device)
            ep.extend(full_step)  # Episode will auto-finalize if done

            if ep.is_complete:
                completed.append(ep.episode_id)
                # immediately start a new active episode and add to dataset
                new_ep = self._new_episode(env_idx)
                self.dataset.add(new_ep)
                self._active[env_idx] = new_ep
                self._expect_first[env_idx] = True  # next digest row = reset obs

        return completed

    def _new_episode(self, env_idx: int) -> Episode:
        ep = Episode(device=self.device)
        if self._episode_metadata_factory is not None:
            try:
                ep.metadata = dict(self._episode_metadata_factory(env_idx, ep.episode_id))
            except TypeError:
                ep.metadata = dict(self._episode_metadata_factory(env_idx))
        return ep

    def _to_full_step(self, row: TensorDict, *, device: Optional[torch.device]) -> TensorDict:
        """Normalize a row to Episode's full-step schema."""
        keys = set(row.keys())
        has_done = (self._done_key in keys) if self._done_key is not None else False
        has_term = (self._terminated_key in keys) if self._terminated_key is not None else False
        has_trunc = (self._truncated_key in keys) if self._truncated_key is not None else False

        if not {"action", "next_obs", "reward"}.issubset(keys):
            missing = {"action", "next_obs", "reward"} - keys
            raise KeyError(f"Missing required keys in step: {missing}")

        if has_done:
            terminated_val = row.get(self._done_key)
            truncated_val = torch.zeros((), dtype=torch.bool, device=terminated_val.device)
        elif has_term and has_trunc:
            terminated_val = row.get(self._terminated_key)
            truncated_val = row.get(self._truncated_key)
        else:
            raise KeyError(
                f"Expected either '{self._done_key}' or both "
                f"'{self._terminated_key}' and '{self._truncated_key}' in step keys: {list(keys)}"
            )

        step = TensorDict({}, batch_size=[], device=device)
        step.set("action", row.get("action").to(device))
        step.set("next_obs", row.get("next_obs").to(device))
        step.set("reward", row.get("reward").to(device))
        step.set("terminated", terminated_val.to(device))
        step.set("truncated", truncated_val.to(device))
        return step
