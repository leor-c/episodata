"""Query and sampling API.

A :class:`Loader` is a declarative description of what the user wants —
fields, segment shape, batch size, filtering — decoupled from how the
backend executes the reads. The v1 execution strategy is straightforward
per-segment reads; a backend-aware planner can replace it later without
changing this API.


Segments are sampled uniformly over all valid (episode, start) pairs. An
episode shorter than the segment still contributes one segment: it is
zero-padded up to the segment length (at the end by default, at the start
with ``pad="prefix"``), and the per-step ``mask`` marks which steps are
real. :class:`SegmentDataset` (a map-style, indexable view — suited to
``torch.utils.data.DataLoader`` and its ``num_workers`` parallelism) is
where segments are read, padded and collated; :class:`Loader` (an
infinite, shuffled, with-replacement stream) is a thin sampling policy on
top of it, drawing random indices and collating ``segments[i]`` items into
batches — one source of truth for segment semantics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Callable

import numpy as np

from .backends.base import Selection, StorageBackend, normalize_payload
from .fields import Fields
from .segment import Batch, Segment

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

    observations: Fields
    actions: Fields
    rewards: np.ndarray | None
    next_observations: Fields
    terminated: np.ndarray
    truncated: np.ndarray


def _resolve_segment_length(
    sequence_length: int | None,
    context_length: int | None,
    target_length: int | None,
) -> int:
    if sequence_length is not None and target_length is not None:
        raise ValueError("pass either sequence_length or context_length/target_length")
    if target_length is not None:
        segment_length = (context_length or 0) + target_length
    elif sequence_length is not None:
        if context_length:
            raise ValueError("context_length requires target_length, not sequence_length")
        segment_length = sequence_length
    else:
        segment_length = 1
    if segment_length < 1:
        raise ValueError("segment length must be >= 1")
    return segment_length


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
    data: dict[str, np.ndarray], length: int, segment_length: int, mode: str | None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Zero-pad a ``length``-step segment read up to ``segment_length`` steps
    and return it with the per-step validity mask (True on real steps)."""
    mask = np.ones(segment_length, dtype=bool)
    pad = segment_length - length
    if pad:
        data = {k: _pad_axis0(v, pad, mode) for k, v in data.items()}
        if mode == "prefix":
            mask[:pad] = False
        else:
            mask[length:] = False
    return data, mask


@dataclasses.dataclass(frozen=True)
class SegmentIndex:
    """Snapshot of valid sampling segments: episode ids, lengths, and the
    cumulative segment count used to map a flat draw to a segment
    :class:`Selection`.

    :class:`SegmentDataset` builds one at construction, giving it the
    stable ``__len__`` a map-style dataset needs, and re-snapshots on
    :meth:`SegmentDataset.refresh` when the backend has seen writes.

    With ``pad`` set, a non-empty episode shorter than the segment counts
    as one segment; ``pad=None`` skips such episodes instead.
    """

    episode_ids: np.ndarray
    lengths: np.ndarray
    cumulative_segments: np.ndarray
    segment_length: int
    pad: str | None

    @classmethod
    def build(
        cls,
        dataset: "Dataset",
        segment_length: int,
        filter: Callable[["Episode"], bool] | None = None,
        pad: str | None = "suffix",
    ) -> "SegmentIndex":
        episode_ids, lengths = [], []
        for episode in dataset.episodes():
            if filter is not None and not filter(episode):
                continue
            length = len(episode)
            if length >= segment_length or (pad is not None and length > 0):
                episode_ids.append(episode.id)
                lengths.append(length)
        lengths_arr = np.asarray(lengths, dtype=np.int64)
        segment_counts = np.maximum(lengths_arr - segment_length + 1, 1)
        return cls(
            episode_ids=np.asarray(episode_ids, dtype=np.int64),
            lengths=lengths_arr,
            cumulative_segments=np.cumsum(segment_counts),
            segment_length=segment_length,
            pad=pad,
        )

    def __len__(self) -> int:
        return int(self.cumulative_segments[-1]) if len(self.cumulative_segments) else 0

    def resolve(self, flat_idx: int) -> tuple[Selection, int | None]:
        """Map a flat segment index to a Selection, plus the in-segment
        offset of the episode's terminal step (``None`` if this segment
        doesn't reach the episode's end).

        For an episode shorter than the segment, the Selection covers the
        whole episode (fewer than ``segment_length`` steps); the caller
        pads the read up to the segment length according to ``pad``."""
        if not 0 <= flat_idx < len(self):
            raise IndexError(flat_idx)
        slot = int(np.searchsorted(self.cumulative_segments, flat_idx, side="right"))
        episode_id = int(self.episode_ids[slot])
        previous = int(self.cumulative_segments[slot - 1]) if slot else 0
        start = int(flat_idx - previous)
        length = int(self.lengths[slot])
        stop = min(start + self.segment_length, length)
        selection = Selection(episode_id, start, stop)
        last = length - 1
        pad_front = self.segment_length - selection.length if self.pad == "prefix" else 0
        terminal_offset = (
            last - start + pad_front if start <= last < start + self.segment_length else None
        )
        return selection, terminal_offset


class SegmentDataset:
    """Map-style, indexable view over fixed-length segments of a dataset.

    The segment index is a snapshot taken at construction — no built-in
    shuffling, no implicit rebuild on online-appended episodes. That's
    what a map-style dataset needs: a stable ``__len__`` and an
    order-independent ``__getitem__``, which is exactly the protocol
    ``torch.utils.data.DataLoader`` uses to shard reads across
    ``num_workers`` worker processes. On a growing dataset, call
    :meth:`refresh` between epochs (never mid-iteration) to make newly
    appended episodes visible::

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

    An episode shorter than the segment yields one segment, zero-padded up
    to the segment length: at the end with ``pad="suffix"`` (default), at
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
        self.segment_length = _resolve_segment_length(sequence_length, context_length, target_length)
        self.pad = _resolve_pad(pad)
        self.filter = filter
        self._revision = dataset.backend.revision
        self._index = SegmentIndex.build(dataset, self.segment_length, filter, pad=self.pad)

    def refresh(self) -> None:
        """Re-snapshot the segment index so episodes appended since
        construction (or the last refresh) become visible.

        Cheap when nothing changed: the index is only rebuilt if the
        backend has seen writes. Never call mid-iteration — ``__len__``
        must stay stable while a ``DataLoader`` epoch is in flight.
        """
        revision = self.dataset.backend.revision
        if revision != self._revision:
            self._index = SegmentIndex.build(
                self.dataset, self.segment_length, self.filter, pad=self.pad
            )
            self._revision = revision

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> Segment:
        selection, terminal_offset = self._index.resolve(i)
        data = read_segment(self.dataset.backend, self.fields, selection)
        data, mask = pad_segment(data, selection.length, self.segment_length, self.pad)
        terminated = np.zeros(self.segment_length, dtype=bool)
        truncated = np.zeros(self.segment_length, dtype=bool)
        if terminal_offset is not None:
            backend = self.dataset.backend
            terminated[terminal_offset] = backend.episode_terminated(selection.episode_id)
            truncated[terminal_offset] = backend.episode_truncated(selection.episode_id)
        return Segment(
            data, self.dataset.schema, terminated=terminated, truncated=truncated, mask=mask
        )

    def collate(self, items: list[Segment]) -> Batch:
        """Combine single segments into a :class:`Batch`. Pass this as
        ``DataLoader``'s ``collate_fn``."""
        data = {k: np.stack([item[k] for item in items], axis=0) for k in self.fields}
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
    """Infinite, shuffled, with-replacement segment stream.

    A thin sampling policy over :class:`SegmentDataset` (exposed as
    ``self.segments``): each :meth:`sample` refreshes the view — a no-op
    unless the dataset grew — draws ``batch_size`` uniform segment
    indices, and collates the indexed segments into a :class:`Batch`.
    """

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
        self.segments = SegmentDataset(
            dataset,
            fields=fields,
            sequence_length=sequence_length,
            context_length=context_length,
            target_length=target_length,
            filter=filter,
            pad=pad,
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    @property
    def dataset(self) -> "Dataset":
        return self.segments.dataset

    @property
    def fields(self) -> list[str]:
        return self.segments.fields

    @property
    def segment_length(self) -> int:
        return self.segments.segment_length

    # -- sampling ---------------------------------------------------------------

    def sample(self) -> Batch:
        """Draw one batch of segments uniformly over all valid segments."""
        segments = self.segments
        segments.refresh()
        if len(segments) == 0:
            if segments.pad is None:
                raise ValueError(
                    f"no episode has length >= {segments.segment_length} (after filtering)"
                )
            raise ValueError("no non-empty episodes to sample from (after filtering)")
        draws = self._rng.integers(len(segments), size=self.batch_size)
        return segments.collate([segments[int(i)] for i in draws])

    def sample_transitions(self) -> TransitionBatch:
        """Draw a batch of single-step transitions ``(s, a, r, s', done)``."""
        if self.segment_length != 2:
            raise ValueError("transition sampling requires sequence_length=2")
        batch = self.sample()
        schema = self.dataset.schema
        obs_keys = [k for k in self.fields if schema.field(k).role == "observation"]
        action_keys = [k for k in self.fields if schema.field(k).role == "action"]
        reward_keys = [k for k in self.fields if schema.field(k).role == "reward"]
        return TransitionBatch(
            observations=Fields({k: batch[k][:, 0] for k in obs_keys}, schema),
            actions=Fields({k: batch[k][:, 1] for k in action_keys}, schema),
            rewards=batch[reward_keys[0]][:, 1] if reward_keys else None,
            next_observations=Fields({k: batch[k][:, 1] for k in obs_keys}, schema),
            terminated=batch.terminated[:, 1],
            truncated=batch.truncated[:, 1],
        )

    def __iter__(self) -> Iterator[Batch]:
        """Shuffled: an endless stream of sampled batches. Unshuffled: one
        deterministic sequential scan over every valid segment."""
        if self.shuffle:
            while True:
                yield self.sample()
        else:
            segments = self.segments
            segments.refresh()
            for start in range(0, len(segments), self.batch_size):
                stop = min(start + self.batch_size, len(segments))
                yield segments.collate([segments[i] for i in range(start, stop)])
