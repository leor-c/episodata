"""Episode views and the online episode writer.

An :class:`Episode` is a logical trajectory, not a physical file — a lazy
view over one episode's temporal fields. Nothing is read until a segment is
requested.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .backends.base import Selection
from .segment import Segment

if TYPE_CHECKING:
    from .dataset import Dataset


class Episode:
    """Lazy view of one logical trajectory.

    Lengths and indices count *transitions* (env steps): an episode that
    took ``T`` ``env.step`` calls has ``len(episode) == T``. The reset row
    that internally precedes the first transition is a storage detail,
    surfaced only as ``observations[0]`` of a segment starting at 0.
    """

    def __init__(self, dataset: "Dataset", episode_id: int):
        self._dataset = dataset
        self.id = episode_id

    @property
    def _backend(self):
        return self._dataset.backend

    def __len__(self) -> int:
        # Storage rows minus the reset row. An episode created bare
        # (new_episode() without observations) has no rows yet: length 0.
        return max(self._backend.episode_length(self.id) - 1, 0)

    @property
    def length(self) -> int:
        return len(self)

    @property
    def terminated(self) -> bool:
        """Whether the episode ended in a terminal state."""
        return self._backend.episode_terminated(self.id)

    @property
    def truncated(self) -> bool:
        return self._backend.episode_truncated(self.id)

    @property
    def ongoing(self) -> bool:
        """True while the episode is still open for online appends."""
        return self._backend.episode_ongoing(self.id)

    def segment(
        self,
        start: int = 0,
        stop: int | None = None,
        fields: list[str] | None = None,
    ) -> Segment:
        """Read transitions ``[start, stop)`` of the selected fields.

        Returns a :class:`Segment` of ``stop - start`` transitions with
        per-transition ``terminated``/``truncated``/``mask`` flags; see the
        :mod:`episodata.segment` docstring for the transition contract.
        """
        length = len(self)
        if stop is None:
            stop = length
        if start < 0:
            start += length
        if stop < 0:
            stop += length
        if not (0 <= start <= stop <= length):
            raise IndexError(f"segment [{start}, {stop}) out of range for length {length}")
        fields = self._dataset._resolve_fields(fields)
        # L transitions live on rows [start, stop]. An episode with no rows
        # at all (bare new_episode(), reset row not yet written) has nothing
        # to read; every segment of it is empty.
        if self._backend.episode_length(self.id) == 0:
            schema = self._dataset.schema
            rows = {
                k: np.zeros((0, *schema.field(k).shape), dtype=schema.field(k).dtype)
                for k in fields
            }
        else:
            payload = self._backend.read_fields(fields, Selection(self.id, start, stop + 1))
            rows = dict(payload)
        terminated = np.zeros(stop - start, dtype=bool)
        truncated = np.zeros(stop - start, dtype=bool)
        if stop == length and stop > start:
            terminated[-1] = self.terminated
            truncated[-1] = self.truncated
        return Segment(
            rows,
            self._dataset.schema,
            terminated=terminated,
            truncated=truncated,
            mask=np.ones(stop - start, dtype=bool),
        )

    def step(self, t: int, fields: list[str] | None = None) -> Segment:
        """Read a single transition; arrays and flags have no leading time
        dim (``observations`` from row ``t``, ``actions``/``rewards``/
        ``next_observations`` from row ``t + 1``)."""
        if t < 0:
            t += len(self)
        segment = self.segment(t, t + 1, fields=fields)
        return Segment(
            segment._rows,
            self._dataset.schema,
            terminated=segment.terminated[0],
            truncated=segment.truncated[0],
            mask=segment.mask[0],
            _squeeze=True,
        )

    def read(self, fields: list[str] | None = None) -> Segment:
        """Read the full episode."""
        return self.segment(0, None, fields=fields)

    def writer(self) -> "EpisodeWriter":
        """Reattach a writer to this episode (must still be ongoing)."""
        return self._dataset.resume_episode(self.id)

    def __repr__(self) -> str:
        return (
            f"Episode(id={self.id}, length={len(self)}, terminated={self.terminated}, "
            f"truncated={self.truncated}, ongoing={self.ongoing})"
        )


class EpisodeWriter:
    """Online append handle for one ongoing episode.

    The writer is stateless: all episode state lives in the backend, keyed
    by ``episode_id``. An equivalent writer can be reattached at any time
    with ``dataset.resume_episode(episode_id)`` or ``episode.writer()``,
    and the same operations exist directly on :class:`Dataset` by id
    (``add_step`` / ``add_steps`` / ``end_episode``).
    """

    def __init__(self, dataset: "Dataset", episode_id: int):
        self._dataset = dataset
        self.episode_id = episode_id
        self._closed = False

    def _write_reset(
        self,
        observations: Mapping[str, Any] | Any,
        infos: Mapping[str, Any] | None = None,
    ) -> None:
        """Write the reset row: initial observation, dummy zero action/reward.

        Normally written by ``Dataset.new_episode(obs, infos)``; this hook
        exists for adapters that receive the reset observation only after
        the writer was created (see ``ActionOutWriter``).
        """
        self._check_open()
        self._dataset._write_reset_row(self.episode_id, observations, infos=infos)

    def add_step(self, step: Mapping[str, Any]) -> None:
        """Append one step; leaf values carry no time dimension.

        A True ``terminated`` / ``truncated`` signal in the step finalizes
        the episode and closes this writer (Gymnasium semantics).
        """
        self._check_open()
        self._dataset.add_step(self.episode_id, step)
        self._sync_closed()

    def add_steps(self, steps: Mapping[str, Any]) -> None:
        """Append a segment; leaf values carry a leading time dimension.

        As with :meth:`add_step`, a True termination signal (only allowed on
        the final step) finalizes the episode and closes this writer.
        """
        self._check_open()
        self._dataset.add_steps(self.episode_id, steps)
        self._sync_closed()

    def _sync_closed(self) -> None:
        if not self._dataset.backend.episode_ongoing(self.episode_id):
            self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("writer is closed")

    def end(self, terminated: bool = False, truncated: bool = False) -> Episode:
        """Finalize the episode and persist its termination flags."""
        episode = self._dataset.end_episode(self.episode_id, terminated, truncated)
        self._closed = True
        return episode

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed and exc_type is None:
            self.end()