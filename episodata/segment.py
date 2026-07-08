"""Temporal field containers: segments and batches of segments.

A :class:`Segment` is a contiguous run of steps — all of its fields
(observations, actions, rewards, infos) plus the per-step episode-boundary
metadata (``terminated``/``truncated``/``mask``). A :class:`Batch` stacks
segments along a leading batch axis.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from .fields import Fields

if TYPE_CHECKING:
    from .schema import DatasetSchema


class Segment(Fields):
    """One unbatched segment: field arrays shaped ``[L, ...]`` (or, for a
    single step, no leading time dim).

    Beyond the :class:`Fields` access patterns (flat keys, spaces, groups,
    role views), a segment carries per-step episode-boundary metadata
    aligned with the leading dims of its fields:

    - ``terminated`` / ``truncated``: True only on the final step of a
      terminated (resp. truncated) episode.
    - ``mask``: True on real steps, False on the zero-padding of a segment
      drawn from a too-short episode.

    Returned by :meth:`Episode.segment` and :class:`SegmentDataset`; combine
    a list of these into a :class:`Batch` via
    :meth:`SegmentDataset.collate`.
    """

    def __init__(
        self,
        data: Mapping[str, np.ndarray],
        schema: "DatasetSchema",
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ):
        super().__init__(data, schema)
        self.terminated = terminated
        self.truncated = truncated
        self.mask = mask

    def select(self, fields: list[str]) -> "Segment":
        return type(self)(
            {k: self._data[k] for k in fields},
            self._schema,
            terminated=self.terminated,
            truncated=self.truncated,
            mask=self.mask,
        )


class Batch(Segment):
    """A batch of segments: field arrays shaped ``[B, L, ...]``, with
    ``terminated``/``truncated``/``mask`` shaped ``[B, L]``.

    When the loader was configured with context/target segments, ``context``
    and ``target`` expose the corresponding time slices.
    """

    def __init__(
        self,
        data: Mapping[str, np.ndarray],
        schema: "DatasetSchema",
        context_length: int | None = None,
        target_length: int | None = None,
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ):
        super().__init__(data, schema, terminated=terminated, truncated=truncated, mask=mask)
        self._context_length = context_length
        self._target_length = target_length

    def select(self, fields: list[str]) -> "Batch":
        return Batch(
            {k: self._data[k] for k in fields},
            self._schema,
            context_length=self._context_length,
            target_length=self._target_length,
            terminated=self.terminated,
            truncated=self.truncated,
            mask=self.mask,
        )

    @property
    def context(self) -> "Batch":
        if not self._context_length:
            raise ValueError("loader was not configured with context_length")
        return self._time_slice(0, self._context_length)

    @property
    def target(self) -> "Batch":
        if self._target_length is None:
            raise ValueError("loader was not configured with target_length")
        start = self._context_length or 0
        return self._time_slice(start, start + self._target_length)

    def _time_slice(self, start: int, stop: int) -> "Batch":
        return Batch(
            {k: v[:, start:stop] for k, v in self._data.items()},
            self._schema,
            terminated=None if self.terminated is None else self.terminated[:, start:stop],
            truncated=None if self.truncated is None else self.truncated[:, start:stop],
            mask=None if self.mask is None else self.mask[:, start:stop],
        )
