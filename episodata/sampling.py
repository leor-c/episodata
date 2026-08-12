"""Query and sampling API.

A :class:`SegmentStream` describes what the user wants — fields, segment
shape, batch size, filtering and sampling policy — independently of how the
backend executes the reads. Every batched query resolves all requested
windows first and sends them to the backend in one call, so storage-specific
overhead is amortized without leaking into the query API.

A segment is a window of *transitions* (see :mod:`episodata.segment` for the
transition contract), and ``sequence_length`` / ``context_length`` /
``target_length`` count transitions. By default, segments are sampled
uniformly over all valid (episode, start) pairs; a custom :class:`Sampler`
can replace that policy. An episode shorter than the segment still contributes
one segment: its row buffer is zero-padded up to the segment length (at the end
by default, at the start with ``pad="prefix"``), and the per-transition
``mask`` marks which transitions are real.
:class:`SegmentDataset` (a map-style, indexable view — suited to
``torch.utils.data.DataLoader`` and its ``num_workers`` parallelism) is
where indices are resolved and segments are read, padded and batched.
:class:`SegmentStream` (an infinite, shuffled stream by default) is a sampling
policy over it, using :meth:`SegmentDataset.fetch` and buffering larger read
chunks — one source of truth for segment semantics and batched access.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Callable

import numpy as np

from .backends.base import Selection
from .sampler import Sampler, UniformSampler
from .segment import Batch, Segment

if TYPE_CHECKING:
    from .dataset import Dataset
    from .episode import Episode
    from .schema import DatasetSchema


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


def pad_segment(
    rows: dict[str, np.ndarray], num_transitions: int, segment_length: int, mode: str | None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Zero-pad the row buffer of a ``num_transitions``-transition read up to
    ``segment_length`` transitions (i.e. ``segment_length + 1`` rows) and
    return it with the per-transition validity mask (True on real
    transitions)."""
    mask = np.ones(segment_length, dtype=bool)
    pad = segment_length - num_transitions
    if pad:
        rows = {k: _pad_axis0(v, pad, mode) for k, v in rows.items()}
        if mode == "prefix":
            mask[:pad] = False
        else:
            mask[num_transitions:] = False
    return rows, mask


def pad_and_stack_segments(
    per_item: Sequence[Mapping[str, np.ndarray]],
    fields: Sequence[str],
    schema: "DatasetSchema",
    segment_length: int,
    mode: str | None,
) -> dict[str, np.ndarray]:
    """Vectorized equivalent of stacking ``pad_segment(item, ...)`` over a
    batch: builds each field's ``[B, L+1, *shape]`` buffer directly by
    scattering each item's (un-padded) rows into a zero-initialized array,
    instead of padding + ``np.stack`` per item.

    Preallocating from the schema (not the read data) means this is correct
    even when every item in the batch happens to be short/padded. The
    per-item scatter loop is pure memory copy, no I/O — negligible next to
    the batched backend read it replaces N of."""
    B = len(per_item)
    out: dict[str, np.ndarray] = {}
    for key in fields:
        spec = schema.field(key)
        buf = np.zeros((B, segment_length + 1, *spec.shape), dtype=spec.dtype)
        for b, item in enumerate(per_item):
            arr = item[key]
            n = arr.shape[0]
            if mode == "prefix":
                buf[b, segment_length + 1 - n :] = arr
            else:
                buf[b, :n] = arr
        out[key] = buf
    return out


@dataclasses.dataclass(frozen=True)
class SegmentIndex:
    """Snapshot of valid sampling segments: episode ids, lengths (in
    transitions), and the cumulative segment count used to map a flat draw
    to a row :class:`Selection`.

    :class:`SegmentDataset` builds one at construction, giving it the
    stable ``__len__`` a map-style dataset needs, and re-snapshots on
    :meth:`SegmentDataset.refresh` when the backend has seen writes.

    With ``pad`` set, an episode with at least one transition but fewer
    than the segment length counts as one segment; ``pad=None`` skips such
    episodes instead. Episodes with no transitions are never sampled.
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
        """Map a flat segment index to a row Selection, plus the in-segment
        offset of the episode's terminal transition (``None`` if this
        segment doesn't reach the episode's end).

        Transitions ``[start, stop)`` live on rows ``[start, stop]``, so the
        Selection covers one row more than the transition count. For an
        episode shorter than the segment, it covers the whole episode; the
        caller pads the read up to the segment length according to
        ``pad``."""
        if not 0 <= flat_idx < len(self):
            raise IndexError(flat_idx)
        slot = int(np.searchsorted(self.cumulative_segments, flat_idx, side="right"))
        episode_id = int(self.episode_ids[slot])
        previous = int(self.cumulative_segments[slot - 1]) if slot else 0
        start = int(flat_idx - previous)
        length = int(self.lengths[slot])
        stop = min(start + self.segment_length, length)
        selection = Selection(episode_id, start, stop + 1)
        last = length - 1
        pad_front = self.segment_length - (stop - start) if self.pad == "prefix" else 0
        terminal_offset = (
            last - start + pad_front if start <= last < start + self.segment_length else None
        )
        return selection, terminal_offset

    def resolve_many(
        self, flat_indices: np.ndarray
    ) -> tuple[list[Selection], list[int | None]]:
        """Vectorized equivalent of calling :meth:`resolve` once per index
        in ``flat_indices`` — one ``np.searchsorted`` call for the whole
        batch instead of one per index, and vectorized arithmetic for the
        rest. Building the actual ``Selection`` objects still needs a
        Python loop (dataclasses aren't vectorizable), but that loop is
        pure attribute assignment, not I/O."""
        flat_indices = np.asarray(flat_indices, dtype=np.int64)
        n = len(self)
        if flat_indices.size and (
            bool(flat_indices.min() < 0) or bool(flat_indices.max() >= n)
        ):
            bad = flat_indices[(flat_indices < 0) | (flat_indices >= n)]
            raise IndexError(int(bad[0]))

        slots = np.searchsorted(self.cumulative_segments, flat_indices, side="right")
        episode_ids = self.episode_ids[slots]
        previous = np.where(slots > 0, self.cumulative_segments[np.clip(slots - 1, 0, None)], 0)
        starts = flat_indices - previous
        lengths = self.lengths[slots]
        stops = np.minimum(starts + self.segment_length, lengths)
        lasts = lengths - 1
        pad_front = self.segment_length - (stops - starts) if self.pad == "prefix" else 0
        has_terminal = (starts <= lasts) & (lasts < starts + self.segment_length)
        terminal_offsets_arr = lasts - starts + pad_front

        selections = [
            Selection(int(episode_ids[i]), int(starts[i]), int(stops[i]) + 1)
            for i in range(len(flat_indices))
        ]
        terminal_offsets = [
            int(terminal_offsets_arr[i]) if has_terminal[i] else None
            for i in range(len(flat_indices))
        ]
        return selections, terminal_offsets


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
    class only implements ``__len__``/``__getitem__``/``__getitems__``,
    which satisfies ``DataLoader``'s map-style protocol by duck typing. It
    works equally well without torch installed at all (e.g. ``segments[i]``
    directly, or your own multiprocessing).

    ``__getitems__`` is ``DataLoader``'s batched-fetch hook: when present,
    ``DataLoader`` calls it once per training batch instead of looping
    ``__getitem__`` once per index, so the ``DataLoader`` usage above is
    fast even on backends (e.g. zarr) where a single-item read has real
    per-call overhead. :meth:`fetch` is the same batched read taken one
    step further — skips per-item ``Segment`` objects entirely and returns
    a stacked ``Batch`` directly; it's what :class:`SegmentStream` uses.

    Segments are windows of transitions and ``sequence_length`` counts
    transitions. An episode shorter than the segment yields one segment,
    zero-padded up to the segment length: at the end with ``pad="suffix"``
    (default), at the start with ``pad="prefix"``. ``Segment.mask`` (and
    ``Batch.mask`` after collation) is True on real transitions.
    ``pad=None`` skips short episodes instead.
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
        self.fields = dataset.schema.resolve_fields(fields)
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
        rows = dict(self.dataset.backend.read_fields(self.fields, [selection])[0])
        rows, mask = pad_segment(rows, selection.length - 1, self.segment_length, self.pad)
        terminated = np.zeros(self.segment_length, dtype=bool)
        truncated = np.zeros(self.segment_length, dtype=bool)
        if terminal_offset is not None:
            backend = self.dataset.backend
            terminated[terminal_offset] = backend.episode_terminated(selection.episode_id)
            truncated[terminal_offset] = backend.episode_truncated(selection.episode_id)
        return Segment(
            rows, self.dataset.schema, terminated=terminated, truncated=truncated, mask=mask
        )

    def __getitems__(self, indices: Sequence[int]) -> list[Segment]:
        """``torch.utils.data.DataLoader``'s batched-fetch hook: when a
        ``Dataset`` defines this, the default fetcher calls it once per
        training batch instead of looping ``__getitem__`` once per index,
        so a plain ``DataLoader(segments, batch_size=N, collate_fn=segments.
        collate)`` gets a faster (one batched backend read, not N) path for
        free. Still returns a list of per-item ``Segment``s — same
        contract ``collate_fn`` already expects — unlike :meth:`fetch`,
        which skips straight to a stacked ``Batch``."""
        selections, terminal_offsets = self._index.resolve_many(np.asarray(indices))
        per_item = self.dataset.backend.read_fields(self.fields, selections)
        backend = self.dataset.backend
        segments = []
        for rows, selection, terminal_offset in zip(per_item, selections, terminal_offsets):
            padded_rows, mask = pad_segment(
                dict(rows), selection.length - 1, self.segment_length, self.pad
            )
            terminated = np.zeros(self.segment_length, dtype=bool)
            truncated = np.zeros(self.segment_length, dtype=bool)
            if terminal_offset is not None:
                terminated[terminal_offset] = backend.episode_terminated(selection.episode_id)
                truncated[terminal_offset] = backend.episode_truncated(selection.episode_id)
            segments.append(
                Segment(
                    padded_rows, self.dataset.schema,
                    terminated=terminated, truncated=truncated, mask=mask,
                )
            )
        return segments

    def fetch(self, indices: Sequence[int]) -> Batch:
        """Vectorized equivalent of ``self.collate(self.__getitems__(indices))``:
        indices in, one batched backend read, a stacked ``Batch`` out — no
        per-item ``Segment`` objects, no per-item ``np.stack``. The
        canonical entry point every batched sampling path (``SegmentStream``,
        ``Dataset.sample_transitions``) routes through."""
        indices = np.asarray(indices)
        selections, terminal_offsets = self._index.resolve_many(indices)
        per_item = self.dataset.backend.read_fields(self.fields, selections)
        rows = pad_and_stack_segments(
            per_item, self.fields, self.dataset.schema, self.segment_length, self.pad
        )

        B = len(indices)
        mask = np.ones((B, self.segment_length), dtype=bool)
        terminated = np.zeros((B, self.segment_length), dtype=bool)
        truncated = np.zeros((B, self.segment_length), dtype=bool)
        backend = self.dataset.backend
        for b, (selection, terminal_offset) in enumerate(zip(selections, terminal_offsets)):
            num_transitions = selection.length - 1
            pad = self.segment_length - num_transitions
            if pad:
                if self.pad == "prefix":
                    mask[b, :pad] = False
                else:
                    mask[b, num_transitions:] = False
            if terminal_offset is not None:
                terminated[b, terminal_offset] = backend.episode_terminated(selection.episode_id)
                truncated[b, terminal_offset] = backend.episode_truncated(selection.episode_id)

        return Batch(
            rows,
            self.dataset.schema,
            context_length=self.context_length,
            target_length=self.target_length,
            terminated=terminated,
            truncated=truncated,
            mask=mask,
        )

    def collate(self, items: list[Segment]) -> Batch:
        """Combine single segments into a :class:`Batch`. Pass this as
        ``DataLoader``'s ``collate_fn``. The row buffers are stacked once;
        the batch's transition views re-derive from the stacked buffer."""
        rows = {k: np.stack([item._rows[k] for item in items], axis=0) for k in self.fields}
        terminated = np.stack([item.terminated for item in items], axis=0)
        truncated = np.stack([item.truncated for item in items], axis=0)
        mask = np.stack([item.mask for item in items], axis=0)
        return Batch(
            rows,
            self.dataset.schema,
            context_length=self.context_length,
            target_length=self.target_length,
            terminated=terminated,
            truncated=truncated,
            mask=mask,
        )


class SegmentStream:
    """Infinite, shuffled, with-replacement segment stream.

    A thin sampling policy over :class:`SegmentDataset` (exposed as
    ``self.segments``): each :meth:`sample` draws indices from ``sampler``
    (uniform with replacement by default — see :class:`~episodata.sampler.
    UniformSampler`) and turns them into a :class:`Batch` via
    :meth:`SegmentDataset.fetch`.

    Internally, ``sample()`` doesn't fetch one ``batch_size`` at a time:
    ``read_chunk_size`` (default ``max(batch_size, 2048)``) controls how
    many segments are drawn and fetched *in one backend call*, buffering
    ``read_chunk_size // batch_size`` ready-to-serve batches before the
    next backend call. This is what actually makes sampling fast on
    backends with real per-call overhead (e.g. zarr): batching to just
    ``batch_size`` barely helps, batching far larger does — see the
    library's benchmarks. The public contract (iterate for ``Batch``es) is
    unchanged; this is purely an internal efficiency knob. One
    consequence: the segment index (see :meth:`SegmentDataset.refresh`) is
    re-snapshotted once per refill, not once per ``sample()`` call, so a
    growing dataset's newest episodes become visible with a bounded delay
    of up to one chunk's worth of batches, not immediately.
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
        sampler: Sampler | None = None,
        read_chunk_size: int | None = None,
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
        self.sampler = sampler if sampler is not None else UniformSampler(seed=seed)
        self.read_chunk_size = (
            read_chunk_size if read_chunk_size is not None else max(batch_size, 2048)
        )
        self._buffer: deque[Batch] = deque()

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

    def _refill(self) -> None:
        segments = self.segments
        segments.refresh()
        if len(segments) == 0:
            if segments.pad is None:
                raise ValueError(
                    f"no episode has length >= {segments.segment_length} (after filtering)"
                )
            raise ValueError("no non-empty episodes to sample from (after filtering)")
        chunk_batches = max(1, self.read_chunk_size // self.batch_size)
        n = chunk_batches * self.batch_size
        indices = self.sampler.sample(segments._index, n)
        batch = segments.fetch(indices)
        for start in range(0, n, self.batch_size):
            stop = start + self.batch_size
            self._buffer.append(batch.map(lambda arr, s=start, e=stop: arr[s:e]))

    def sample(self) -> Batch:
        """Draw one batch of segments (uniform, or per ``sampler``, over
        all valid segments). See the class docstring for the internal
        chunked-fetch/staleness trade-off."""
        if not self._buffer:
            self._refill()
        return self._buffer.popleft()

    def sample_transitions(self) -> Batch:
        """Draw a batch of single transitions ``(s, a, r, s', done)`` — a
        length-1 segment batch with the time dim squeezed away, so every
        role accessor is ``[B, ...]``: ``t.obs``, ``t.action``, ``t.reward``,
        ``t.next_obs``, ``t.terminated``."""
        if self.segment_length != 1:
            raise ValueError("transition sampling requires sequence_length=1")
        batch = self.sample()
        return Batch(
            batch._rows,
            self.dataset.schema,
            terminated=batch.terminated[:, 0],
            truncated=batch.truncated[:, 0],
            mask=batch.mask[:, 0],
            _squeeze=True,
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
                yield segments.fetch(np.arange(start, stop))
