"""episodata — a unified episode dataset library for world models and control.

Stable semantics (schema) + use-case-driven queries (loader) + replaceable
storage (backends). See ``world_model_dataset_design.md``.
"""

from .action_out import ActionOutWriter
from .backends import (
    MemoryBackend,
    NpzDirectoryBackend,
    Selection,
    SpaceBlock,
    StorageBackend,
    get_backend,
    register_backend,
)
from .dataset import Dataset
from .episode import Episode, EpisodeWriter
from .loader import Loader, Segment, SegmentDataset, TransitionBatch
from .observation import Batch, FieldGroup, Observation, SpaceView
from .schema import DatasetSchema, FieldSpec, SpaceSpec

__version__ = "0.1.0"

__all__ = [
    "ActionOutWriter",
    "Batch",
    "Dataset",
    "DatasetSchema",
    "Episode",
    "EpisodeWriter",
    "FieldGroup",
    "FieldSpec",
    "Loader",
    "MemoryBackend",
    "NpzDirectoryBackend",
    "Observation",
    "Segment",
    "SegmentDataset",
    "Selection",
    "SpaceBlock",
    "SpaceSpec",
    "SpaceView",
    "StorageBackend",
    "TransitionBatch",
    "get_backend",
    "register_backend",
]