"""Temporal field containers: segments and batches of segments.

A :class:`Segment` is a contiguous run of *transitions* — the only read
product episodata has. Every field is named for its place in the transition,
so there is no alignment convention to know:

- ``observations`` / ``infos``: the observation each action was taken at;
  for a window starting at the episode's beginning, ``observations[0]`` is
  the reset observation.
- ``actions`` / ``rewards``: ``actions[i]`` is the action taken at
  ``observations[i]``; ``rewards[i]`` is its reward.
- ``next_observations`` / ``next_infos``: the observation each action
  produced.
- ``terminated`` / ``truncated`` / ``mask``: per-transition flags —
  ``terminated[i]`` is the done signal of transition ``i``.

A window of ``L`` transitions is backed by one buffer of ``L + 1`` storage
rows; all of the above are zero-copy numpy views into it (``observations``
is rows ``[:-1]``, everything else rows ``[1:]``), so e.g. pixel
observations are never duplicated between ``observations`` and
``next_observations``. A :class:`Batch` stacks segments along a leading
batch axis and re-derives the same views from a ``[B, L+1]`` buffer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from .fields import Fields

if TYPE_CHECKING:
    from .schema import DatasetSchema

#: Roles whose data sits on the observation side of a transition (read from
#: the current row); action/reward data is read from the next row.
_ROW_ROLES = ("observation", "info")


class Segment(Fields):
    """One unbatched segment of ``L`` transitions: field arrays shaped
    ``[L, ...]`` (or, for a single transition, no leading time dim).

    Beyond the :class:`Fields` access patterns (flat keys, spaces, groups,
    role views), a segment carries ``next_observations`` / ``next_infos``
    and the per-transition flags ``terminated`` / ``truncated`` / ``mask``
    (True on real transitions, False on the zero-padding of a segment drawn
    from a too-short episode). See the module docstring for the transition
    contract.

    Returned by :meth:`Episode.segment` and :class:`SegmentDataset`; combine
    a list of these into a :class:`Batch` via
    :meth:`SegmentDataset.collate`.
    """

    #: Axis of the row buffer that indexes time (1 in :class:`Batch`).
    _time_axis = 0

    def __init__(
        self,
        rows: Mapping[str, np.ndarray],
        schema: "DatasetSchema",
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        *,
        _squeeze: bool = False,
    ):
        self._rows = dict(rows)
        self._squeeze = _squeeze
        current = 0 if _squeeze else slice(None, -1)
        shifted = 1 if _squeeze else slice(1, None)
        data: dict[str, np.ndarray] = {}
        self._next: dict[str, np.ndarray] = {}
        for key, arr in self._rows.items():
            shifted_view = self._take(arr, shifted)
            if schema.fields[key].role in _ROW_ROLES:
                data[key] = self._take(arr, current)
                self._next[key] = shifted_view
            else:
                data[key] = shifted_view
        super().__init__(data, schema)
        self.terminated = terminated
        self.truncated = truncated
        self.mask = mask

    def _take(self, arr: np.ndarray, index) -> np.ndarray:
        return arr[(slice(None),) * self._time_axis + (index,)]

    @property
    def next_observations(self) -> Fields:
        """Sub-view of the observation each transition's action produced."""
        return Fields(self._next, self._schema).observations

    @property
    def next_infos(self) -> Fields:
        """Sub-view of the info paired with each ``next_observation``."""
        return Fields(self._next, self._schema).infos

    def select(self, fields: list[str]) -> "Segment":
        return type(self)(
            {k: self._rows[k] for k in fields},
            self._schema,
            terminated=self.terminated,
            truncated=self.truncated,
            mask=self.mask,
            _squeeze=self._squeeze,
        )


class Batch(Segment):
    """A batch of segments: field arrays shaped ``[B, L, ...]``, with
    ``terminated``/``truncated``/``mask`` shaped ``[B, L]`` — all views into
    one ``[B, L+1]`` row buffer, per the transition contract in the module
    docstring.

    When the loader was configured with context/target segments, ``context``
    and ``target`` expose the corresponding transition slices; they share
    one boundary row, so ``target.observations`` starts exactly where
    ``context.next_observations`` ends.
    """

    _time_axis = 1

    def __init__(
        self,
        rows: Mapping[str, np.ndarray],
        schema: "DatasetSchema",
        context_length: int | None = None,
        target_length: int | None = None,
        terminated: np.ndarray | None = None,
        truncated: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ):
        super().__init__(rows, schema, terminated=terminated, truncated=truncated, mask=mask)
        self._context_length = context_length
        self._target_length = target_length

    def select(self, fields: list[str]) -> "Batch":
        return Batch(
            {k: self._rows[k] for k in fields},
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
        # Transitions [start, stop) live on rows [start, stop]: adjacent
        # slices share their boundary row.
        return Batch(
            {k: v[:, start : stop + 1] for k, v in self._rows.items()},
            self._schema,
            terminated=None if self.terminated is None else self.terminated[:, start:stop],
            truncated=None if self.truncated is None else self.truncated[:, start:stop],
            mask=None if self.mask is None else self.mask[:, start:stop],
        )
