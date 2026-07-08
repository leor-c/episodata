"""episodata — a unified episode dataset library for world models and control.

Stable semantics (schema) + use-case-driven queries (sampling) + replaceable
storage (backends). See ``world_model_dataset_design.md``.
"""

from .action_out import ActionOutWriter
from .backends import (
    MemoryBackend,
    NpzDirectoryBackend,
    Selection,
    SpaceBlock,
    StorageBackend,
    ZarrBackend,
    get_backend,
    register_backend,
)
from .dataset import Dataset
from .episode import Episode, EpisodeWriter
from .fields import FieldGroup, Fields, Observation, SpaceView
from .sampling import Loader, SegmentDataset, TransitionBatch
from .schema import DatasetSchema, FieldSpec, SpaceSpec
from .segment import Batch, Segment

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
    "Fields",
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
    "ZarrBackend",
    "get_backend",
    "register_backend",
]