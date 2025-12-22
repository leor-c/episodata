"""Episodata: episode and segment utilities (TensorDict-friendly)."""

__all__ = [
    "Episode",
    "EpisodeDataset",
    "SegmentsDataset",
    "Collector",
]

__version__ = "0.1.0"

from .episode import Episode
from .dataset import EpisodeDataset, SegmentsDataset
from .collector import Collector
