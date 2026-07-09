"""episodata — a unified episode dataset library for world models and control.

Stable semantics (schema) + use-case-driven queries (sampling) + replaceable
storage (backends). See ``docs/api.md`` for the module layering and key
signatures.
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
from .fields import FieldGroup, Fields, SpaceView
from .sampling import SegmentDataset, SegmentStream
from .schema import DatasetSchema, FieldSpec, SpaceSpec
from .segment import Batch, Segment
from .vector import VectorWriter

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
    "MemoryBackend",
    "NpzDirectoryBackend",
    "Segment",
    "SegmentDataset",
    "SegmentStream",
    "Selection",
    "SpaceBlock",
    "SpaceSpec",
    "SpaceView",
    "StorageBackend",
    "VectorWriter",
    "ZarrBackend",
    "get_backend",
    "register_backend",
]