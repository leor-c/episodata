"""Query and sampling API.

A :class:`Loader` is a declarative description of what the user wants —
fields, window shape, batch size, filtering — decoupled from how the
backend executes the reads. The v1 execution strategy is straightforward
per-window reads; a backend-aware planner can replace it later without
changing this API.

Windows are sampled uniformly over all valid (episode, start) pairs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import TYPE_CHECKING, Callable

import numpy as np

from .backends.base import Selection, normalize_payload
from .observation import Batch, Observation

if TYPE_CHECKING:
    from .dataset import Dataset
    from .episode import Episode


@dataclasses.dataclass
class TransitionBatch:
    """A batch of ``(s, a, r, s', done)`` transitions, arrays batched along axis 0.

    Under the action-in convention, ``actions`` and ``rewards`` are the ones
    that led from ``observations`` to ``next_observations`` (stored on row
    ``t + 1``); the pairing is done here, so consumers are alignment-free.
    """

    observations: Observation
    actions: Observation
    rewards: np.ndarray | None
    next_observations: Observation
    terminated: np.ndarray
    truncated: np.ndarray


@dataclasses.dataclass(frozen=True)
class _Index:
    """Snapshot of valid sampling windows: episode ids, lengths, and the
    cumulative window count used to map a flat draw to (episode, start)."""

    episode_ids: np.ndarray
    lengths: np.ndarray
    cumulative_windows: np.ndarray

    @property
    def total_windows(self) -> int:
        return int(self.cumulative_windows[-1]) if len(self.cumulative_windows) else 0


class Loader:
    def __init__(
        self,
        dataset: "Dataset",
        fields: list[str] | None = None,
        batch_size: int = 1,
        sequence_length: int | None = None,
        context_length: int | None = None,
        target_length: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        filter: Callable[["Episode"], bool] | None = None,
    ):
        if sequence_length is not None and target_length is not None:
            raise ValueError("pass either sequence_length or context_length/target_length")
        if target_length is not None:
            self.window = (context_length or 0) + target_length
        elif sequence_length is not None:
            if context_length:
                raise ValueError("context_length requires target_length, not sequence_length")
            self.window = sequence_length
        else:
            self.window = 1
        if self.window < 1:
            raise ValueError("window length must be >= 1")

        self.dataset = dataset
        self.fields = dataset._resolve_fields(fields)
        self.batch_size = batch_size
        self.context_length = context_length
        self.target_length = target_length
        self.shuffle = shuffle
        self.filter = filter
        self._rng = np.random.default_rng(seed)

    # -- index ---------------------------------------------------------------

    def _build_index(self) -> _Index:
        """Snapshot valid windows. Rebuilt per sample so that episodes
        appended online become visible; cheap relative to reads at v1 scale."""
        episode_ids, lengths = [], []
        for episode in self.dataset.episodes():
            if self.filter is not None and not self.filter(episode):
                continue
            length = len(episode)
            if length >= self.window:
                episode_ids.append(episode.id)
                lengths.append(length)
        lengths_arr = np.asarray(lengths, dtype=np.int64)
        windows = lengths_arr - self.window + 1
        return _Index(
            episode_ids=np.asarray(episode_ids, dtype=np.int64),
            lengths=lengths_arr,
            cumulative_windows=np.cumsum(windows),
        )

    # -- sampling ---------------------------------------------------------------

    def sample(self) -> Batch:
        """Draw one batch of windows uniformly over all valid windows."""
        index = self._build_index()
        if index.total_windows == 0:
            raise ValueError(
                f"no episode has length >= {self.window} (after filtering)"
            )
        draws = self._rng.integers(index.total_windows, size=self.batch_size)
        return self._read_batch(index, draws)

    def sample_transitions(self) -> TransitionBatch:
        """Draw a batch of single-step transitions ``(s, a, r, s', done)``."""
        if self.window != 2:
            raise ValueError("transition sampling requires sequence_length=2")
        batch = self.sample()
        schema = self.dataset.schema
        obs_keys = [k for k in self.fields if schema.field(k).role == "observation"]
        action_keys = [k for k in self.fields if schema.field(k).role == "action"]
        reward_keys = [k for k in self.fields if schema.field(k).role == "reward"]
        return TransitionBatch(
            observations=Observation({k: batch[k][:, 0] for k in obs_keys}, schema),
            actions=Observation({k: batch[k][:, 1] for k in action_keys}, schema),
            rewards=batch[reward_keys[0]][:, 1] if reward_keys else None,
            next_observations=Observation({k: batch[k][:, 1] for k in obs_keys}, schema),
            terminated=batch.terminated[:, 1],
            truncated=batch.truncated[:, 1],
        )

    def __iter__(self) -> Iterator[Batch]:
        """Shuffled: an endless stream of sampled batches. Unshuffled: one
        deterministic sequential scan over every valid window."""
        if self.shuffle:
            while True:
                yield self.sample()
        else:
            index = self._build_index()
            all_draws = np.arange(index.total_windows)
            for i in range(0, len(all_draws), self.batch_size):
                yield self._read_batch(index, all_draws[i : i + self.batch_size])

    # -- execution ---------------------------------------------------------------

    def _read_batch(self, index: _Index, draws: np.ndarray) -> Batch:
        backend = self.dataset.backend
        schema = self.dataset.schema
        columns: dict[str, list[np.ndarray]] = {k: [] for k in self.fields}
        terminated = np.zeros((len(draws), self.window), dtype=bool)
        truncated = np.zeros((len(draws), self.window), dtype=bool)

        for row, draw in enumerate(draws):
            slot = int(np.searchsorted(index.cumulative_windows, draw, side="right"))
            episode_id = int(index.episode_ids[slot])
            previous = int(index.cumulative_windows[slot - 1]) if slot else 0
            start = int(draw - previous)
            selection = Selection(episode_id, start, start + self.window)
            payload = normalize_payload(backend.read_fields(self.fields, selection))
            for key in self.fields:
                columns[key].append(payload[key])
            last = int(index.lengths[slot]) - 1
            if start <= last < start + self.window:
                terminated[row, last - start] = backend.episode_terminated(episode_id)
                truncated[row, last - start] = backend.episode_truncated(episode_id)

        data = {k: np.stack(v, axis=0) for k, v in columns.items()}
        return Batch(
            data,
            schema,
            context_length=self.context_length,
            target_length=self.target_length,
            terminated=terminated,
            truncated=truncated,
        )