"""Query and sampling API.

A :class:`Loader` is a declarative description of what the user wants —
fields, window shape, batch size, filtering — decoupled from how the
backend executes the reads. The v1 execution strategy is straightforward
per-window reads; a backend-aware planner can replace it later without
changing this API.

Windows are sampled uniformly over all valid (episode, start) pairs. An
episode shorter than the window still contributes one window: it is
zero-padded up to the window length (at the end by default, at the start
with ``pad="prefix"``), and the per-step ``mask`` marks which steps are
real. Both
:class:`Loader` (an infinite, shuffled, with-replacement stream) and
:class:`SegmentDataset` (a map-style, indexable view — suited to
``torch.utils.data.DataLoader`` and its ``num_workers`` parallelism) are
built on the same window index (:class:`SegmentIndex`) and the same
single-segment read (:func:`read_segment`), so the two sampling modes
share one source of truth instead of drifting apart.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Callable

import numpy as np

from .backends.base import Selection, StorageBackend, normalize_payload
from .observation import Batch, Observation

if TYPE_CHECKING:
    from .dataset import Dataset
    from .episode import Episode
    from .schema import DatasetSchema


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


def _resolve_window(
    sequence_length: int | None,
    context_length: int | None,
    target_length: int | None,
) -> int:
    if sequence_length is not None and target_length is not None:
        raise ValueError("pass either sequence_length or context_length/target_length")
    if target_length is not None:
        window = (context_length or 0) + target_length
    elif sequence_length is not None:
        if context_length:
            raise ValueError("context_length requires target_length, not sequence_length")
        window = sequence_length
    else:
        window = 1
    if window < 1:
        raise ValueError("window length must be >= 1")
    return window


def _resolve_pad(pad: str | None) -> str | None:
    if pad not in ("suffix", "prefix", None):
        raise ValueError(f"pad must be 'suffix', 'prefix' or None, got {pad!r}")
    return pad


def _pad_axis0(arr: np.ndarray, pad: int, mode: str) -> np.ndarray:
    """Zero-pad ``arr`` along axis 0, after ("suffix") or before ("prefix")."""
    widths = [(pad, 0) if mode == "prefix" else (0, pad)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, widths)


def read_segment(
    backend: StorageBackend, fields: Sequence[str], selection: Selection
) -> dict[str, np.ndarray]:
    """Read one segment's fields from the backend, normalized to a flat dict."""
    return normalize_payload(backend.read_fields(fields, selection))


def pad_segment(
    data: dict[str, np.ndarray], length: int, window: int, mode: str | None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Zero-pad a ``length``-step segment read up to ``window`` steps and
    return it with the per-step validity mask (True on real steps)."""
    mask = np.ones(window, dtype=bool)
    pad = window - length
    if pad:
        data = {k: _pad_axis0(v, pad, mode) for k, v in data.items()}
        if mode == "prefix":
            mask[:pad] = False
        else:
            mask[length:] = False
    return data, mask


@dataclasses.dataclass(frozen=True)
class SegmentIndex:
    """Snapshot of valid sampling windows: episode ids, lengths, and the
    cumulative window count used to map a flat draw to a segment
    :class:`Selection`.

    :class:`Loader` rebuilds one per sample, so online-appended episodes
    become visible immediately. :class:`SegmentDataset` builds one at
    construction, giving it the stable ``__len__`` a map-style dataset
    needs.

    With ``pad`` set, a non-empty episode shorter than the window counts
    as one window; ``pad=None`` skips such episodes instead.
    """

    episode_ids: np.ndarray
    lengths: np.ndarray
    cumulative_windows: np.ndarray
    window: int
    pad: str | None

    @classmethod
    def build(
        cls,
        dataset: "Dataset",
        window: int,
        filter: Callable[["Episode"], bool] | None = None,
        pad: str | None = "suffix",
    ) -> "SegmentIndex":
        episode_ids, lengths = [], []
        for episode in dataset.episodes():
            if filter is not None and not filter(episode):
                continue
            length = len(episode)
            if length >= window or (pad is not None and length > 0):
                episode_ids.append(episode.id)
                lengths.append(length)
        lengths_arr = np.asarray(lengths, dtype=np.int64)
        windows = np.maximum(lengths_arr - window + 1, 1)
        return cls(
            episode_ids=np.asarray(episode_ids, dtype=np.int64),
            lengths=lengths_arr,
            cumulative_windows=np.cumsum(windows),
            window=window,
            pad=pad,
        )

    def __len__(self) -> int:
        return int(self.cumulative_windows[-1]) if len(self.cumulative_windows) else 0

    def resolve(self, flat_idx: int) -> tuple[Selection, int | None]:
        """Map a flat window index to a Selection, plus the in-window offset
        of the episode's terminal step (``None`` if this window doesn't
        reach the episode's end).

        For an episode shorter than the window, the Selection covers the
        whole episode (fewer than ``window`` steps); the caller pads the
        read up to the window length according to ``pad``."""
        if not 0 <= flat_idx < len(self):
            raise IndexError(flat_idx)
        slot = int(np.searchsorted(self.cumulative_windows, flat_idx, side="right"))
        episode_id = int(self.episode_ids[slot])
        previous = int(self.cumulative_windows[slot - 1]) if slot else 0
        start = int(flat_idx - previous)
        length = int(self.lengths[slot])
        stop = min(start + self.window, length)
        selection = Selection(episode_id, start, stop)
        last = length - 1
        pad_front = self.window - selection.length if self.pad == "prefix" else 0
        terminal_offset = (
            last - start + pad_front if start <= last < start + self.window else None
        )
        return selection, terminal_offset


@dataclasses.dataclass
class Segment:
    """One unbatched segment: field arrays shaped ``[L, ...]``, with
    per-step ``terminated``/``truncated`` flags and a per-step ``mask``
    (True on real steps, False on padding).

    Returned by :class:`SegmentDataset`; combine a list of these into a
    :class:`Batch` via :meth:`SegmentDataset.collate`.
    """

    data: dict[str, np.ndarray]
    schema: "DatasetSchema"
    terminated: np.ndarray
    truncated: np.ndarray
    mask: np.ndarray

    @property
    def observation(self) -> Observation:
        return Observation(self.data, self.schema)


class SegmentDataset:
    """Map-style, indexable view over fixed-length segments of a dataset.

    Unlike :class:`Loader`, the segment index is a snapshot taken at
    construction — no built-in shuffling, no rebuild on online-appended
    episodes. That's what a map-style dataset needs: a stable ``__len__``
    and an order-independent ``__getitem__``, which is exactly the
    protocol ``torch.utils.data.DataLoader`` uses to shard reads across
    ``num_workers`` worker processes::

        from torch.utils.data import DataLoader

        segments = dataset.segments(fields=[...], sequence_length=8)
        loader = DataLoader(
            segments, batch_size=32, shuffle=True,
            num_workers=4, collate_fn=segments.collate,
        )
        for batch in loader: ...  # episodata.Batch, arrays [B, L, ...]

    No torch import happens here or anywhere in the core library — this
    class only implements ``__len__``/``__getitem__``, which satisfies
    ``DataLoader``'s map-style protocol by duck typing. It works equally
    well without torch installed at all (e.g. ``segments[i]`` directly, or
    your own multiprocessing).

    An episode shorter than the window yields one segment, zero-padded up
    to the window length: at the end with ``pad="suffix"`` (default), at
    the start with ``pad="prefix"``. ``Segment.mask`` (and ``Batch.mask``
    after collation) is True on real steps. ``pad=None`` skips short
    episodes instead.
    """

    def __init__(
        self,
        dataset: "Dataset",
        fields: list[str] | None = None,
        sequence_length: int | None = None,
        context_length: int | None = None,
        target_length: int | None = None,
        filter: Callable[["Episode"], bool] | None = None,
        pad: str | None = "suffix",
    ):
        self.dataset = dataset
        self.fields = dataset._resolve_fields(fields)
        self.context_length = context_length
        self.target_length = target_length
        self.window = _resolve_window(sequence_length, context_length, target_length)
        self.pad = _resolve_pad(pad)
        self._index = SegmentIndex.build(dataset, self.window, filter, pad=self.pad)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Segment:
        selection, terminal_offset = self._index.resolve(i)
        data = read_segment(self.dataset.backend, self.fields, selection)
        data, mask = pad_segment(data, selection.length, self.window, self.pad)
        terminated = np.zeros(self.window, dtype=bool)
        truncated = np.zeros(self.window, dtype=bool)
        if terminal_offset is not None:
            backend = self.dataset.backend
            terminated[terminal_offset] = backend.episode_terminated(selection.episode_id)
            truncated[terminal_offset] = backend.episode_truncated(selection.episode_id)
        return Segment(data, self.dataset.schema, terminated, truncated, mask)

    def collate(self, items: list[Segment]) -> Batch:
        """Combine single segments into a :class:`Batch`. Pass this as
        ``DataLoader``'s ``collate_fn``."""
        data = {k: np.stack([item.data[k] for item in items], axis=0) for k in self.fields}
        terminated = np.stack([item.terminated for item in items], axis=0)
        truncated = np.stack([item.truncated for item in items], axis=0)
        mask = np.stack([item.mask for item in items], axis=0)
        return Batch(
            data,
            self.dataset.schema,
            context_length=self.context_length,
            target_length=self.target_length,
            terminated=terminated,
            truncated=truncated,
            mask=mask,
        )


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
        pad: str | None = "suffix",
    ):
        self.window = _resolve_window(sequence_length, context_length, target_length)
        self.dataset = dataset
        self.fields = dataset._resolve_fields(fields)
        self.batch_size = batch_size
        self.context_length = context_length
        self.target_length = target_length
        self.shuffle = shuffle
        self.filter = filter
        self.pad = _resolve_pad(pad)
        self._rng = np.random.default_rng(seed)

    # -- index ---------------------------------------------------------------

    def _build_index(self) -> SegmentIndex:
        """Snapshot valid windows. Rebuilt per sample so that episodes
        appended online become visible; cheap relative to reads at v1 scale."""
        return SegmentIndex.build(self.dataset, self.window, self.filter, pad=self.pad)

    # -- sampling ---------------------------------------------------------------

    def sample(self) -> Batch:
        """Draw one batch of windows uniformly over all valid windows."""
        index = self._build_index()
        if len(index) == 0:
            if self.pad is None:
                raise ValueError(
                    f"no episode has length >= {self.window} (after filtering)"
                )
            raise ValueError("no non-empty episodes to sample from (after filtering)")
        draws = self._rng.integers(len(index), size=self.batch_size)
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
            all_draws = np.arange(len(index))
            for i in range(0, len(all_draws), self.batch_size):
                yield self._read_batch(index, all_draws[i : i + self.batch_size])

    # -- execution ---------------------------------------------------------------

    def _read_batch(self, index: SegmentIndex, draws: np.ndarray) -> Batch:
        backend = self.dataset.backend
        schema = self.dataset.schema
        columns: dict[str, list[np.ndarray]] = {k: [] for k in self.fields}
        terminated = np.zeros((len(draws), self.window), dtype=bool)
        truncated = np.zeros((len(draws), self.window), dtype=bool)
        mask = np.ones((len(draws), self.window), dtype=bool)

        for row, draw in enumerate(draws):
            selection, terminal_offset = index.resolve(int(draw))
            payload = read_segment(backend, self.fields, selection)
            payload, mask[row] = pad_segment(payload, selection.length, self.window, self.pad)
            for key in self.fields:
                columns[key].append(payload[key])
            if terminal_offset is not None:
                terminated[row, terminal_offset] = backend.episode_terminated(selection.episode_id)
                truncated[row, terminal_offset] = backend.episode_truncated(selection.episode_id)

        data = {k: np.stack(v, axis=0) for k, v in columns.items()}
        return Batch(
            data,
            schema,
            context_length=self.context_length,
            target_length=self.target_length,
            terminated=terminated,
            truncated=truncated,
            mask=mask,
        )