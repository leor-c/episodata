"""Temporal field containers: segments and batches of segments.

A :class:`Segment` is a contiguous run of *transitions* — the only read
product episodata has. Every field is named for its place in the transition,
so there is no alignment convention to know:

- ``observation`` / ``info``: the observation each action was taken at;
  for a window starting at the episode's beginning, ``observation[0]`` is
  the reset observation.
- ``action`` / ``reward``: ``action[i]`` is the action taken at
  ``observation[i]``; ``reward[i]`` is its reward.
- ``next_observation`` / ``next_info``: the observation each action
  produced.
- ``terminated`` / ``truncated`` / ``mask``: per-transition flags —
  ``terminated[i]`` is the done signal of transition ``i``.

Each role accessor returns the bare ``[L, ...]`` array when the source was
a bare array, or a :class:`Fields` view when it was a dict (see
:func:`episodata.fields.role_view`). Singular, plural and the ``obs``
shorthand are aliases for the same object: ``seg.observation`` ==
``seg.obs`` == ``seg.observations``.

A window of ``L`` transitions is backed by one buffer of ``L + 1`` storage
rows; all of the above are zero-copy numpy views into it (``observation``
is rows ``[:-1]``, everything else rows ``[1:]``), so e.g. pixel
observations are never duplicated between ``observation`` and
``next_observation``. A :class:`Batch` stacks segments along a leading
batch axis and re-derives the same views from a ``[B, L+1]`` buffer.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from .fields import Fields, role_view

if TYPE_CHECKING:
    from .schema import DatasetSchema

#: Roles whose data sits on the observation side of a transition (read from
#: the current row); action/reward data is read from the next row.
_ROW_ROLES = ("observation", "info")


class Segment:
    """One unbatched segment of ``L`` transitions: field arrays shaped
    ``[L, ...]`` (or, for a single transition, no leading time dim).

    Access is strictly role-first: ``seg.observation`` / ``seg.obs`` /
    ``seg.observations`` (aliases of one object), ``seg.action``,
    ``seg.reward``, ``seg.info``, plus ``seg.next_observation`` /
    ``seg.next_info`` and the per-transition flags ``terminated`` /
    ``truncated`` / ``mask`` (True on real transitions, False on the
    zero-padding of a segment drawn from a too-short episode). See the
    module docstring for the transition contract.

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
        self._data = data
        self._schema = schema
        self.terminated = terminated
        self.truncated = truncated
        self.mask = mask

    def _take(self, arr: np.ndarray, index) -> np.ndarray:
        return arr[(slice(None),) * self._time_axis + (index,)]

    @property
    def schema(self) -> "DatasetSchema":
        return self._schema

    # -- role accessors ------------------------------------------------------
    # Cached so the aliases below hand back the identical object.

    @cached_property
    def observation(self) -> np.ndarray | Fields:
        """The observation each transition's action was taken at."""
        return role_view(self._data, self._schema, "observation")

    @cached_property
    def action(self) -> np.ndarray | Fields:
        """The action of each transition."""
        return role_view(self._data, self._schema, "action")

    @cached_property
    def reward(self) -> np.ndarray | Fields:
        """The reward of each transition."""
        return role_view(self._data, self._schema, "reward")

    @cached_property
    def info(self) -> np.ndarray | Fields:
        """The info paired with each ``observation``."""
        return role_view(self._data, self._schema, "info")

    @cached_property
    def next_observation(self) -> np.ndarray | Fields:
        """The observation each transition's action produced."""
        return role_view(self._next, self._schema, "observation")

    @cached_property
    def next_info(self) -> np.ndarray | Fields:
        """The info paired with each ``next_observation``."""
        return role_view(self._next, self._schema, "info")

    # -- aliases -------------------------------------------------------------

    @property
    def obs(self):
        return self.observation

    @property
    def observations(self):
        return self.observation

    @property
    def actions(self):
        return self.action

    @property
    def rewards(self):
        return self.reward

    @property
    def infos(self):
        return self.info

    @property
    def next_obs(self):
        return self.next_observation

    @property
    def next_observations(self):
        return self.next_observation

    @property
    def next_infos(self):
        return self.next_info

    def select(self, fields: list[str]) -> "Segment":
        return type(self)(
            {k: self._rows[k] for k in fields},
            self._schema,
            terminated=self.terminated,
            truncated=self.truncated,
            mask=self.mask,
            _squeeze=self._squeeze,
        )

    def __repr__(self) -> str:
        shapes = {k: tuple(v.shape) for k, v in self._data.items()}
        return f"{type(self).__name__}({shapes})"


class Batch(Segment):
    """A batch of segments: field arrays shaped ``[B, L, ...]``, with
    ``terminated``/``truncated``/``mask`` shaped ``[B, L]`` — all views into
    one ``[B, L+1]`` row buffer, per the transition contract in the module
    docstring.

    When the loader was configured with context/target segments, ``context``
    and ``target`` expose the corresponding transition slices; they share
    one boundary row, so ``target.observation`` starts exactly where
    ``context.next_observation`` ends.
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
        *,
        _squeeze: bool = False,
    ):
        super().__init__(
            rows,
            schema,
            terminated=terminated,
            truncated=truncated,
            mask=mask,
            _squeeze=_squeeze,
        )
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
            _squeeze=self._squeeze,
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
