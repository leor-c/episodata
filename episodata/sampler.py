"""Pluggable index generation for segment sampling.

A :class:`Sampler` decides which flat segment indices to draw from a
:class:`~episodata.sampling.SegmentIndex`; :class:`~episodata.sampling.
SegmentDataset` (via :meth:`~episodata.sampling.SegmentDataset.fetch`)
handles turning those indices into data, uniformly whether the underlying
storage is in-memory or persistent. This mirrors torchrl's Sampler/Storage
split: the sampling policy (which indices) and data access (how those
indices become arrays) are independent concerns, so a new sampling policy
(e.g. staleness-aware) never needs to touch how data is read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .sampling import SegmentIndex


class Sampler(Protocol):
    def sample(self, index: "SegmentIndex", batch_size: int) -> np.ndarray:
        """Return ``batch_size`` flat segment indices to draw from ``index``."""
        ...


class UniformSampler:
    """Uniform sampling with replacement — the default, matching
    :class:`~episodata.sampling.SegmentStream`'s original behavior."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def sample(self, index: "SegmentIndex", batch_size: int) -> np.ndarray:
        return self._rng.integers(len(index), size=batch_size)
