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
    StorageBackend,
    ZarrBackend,
    get_backend,
    register_backend,
)
from .dataset import Dataset
from .episode import Episode, EpisodeWriter
from .fields import FieldGroup, Fields
from .sampler import Sampler, UniformSampler
from .sampling import SegmentDataset, SegmentStream
from .schema import DatasetSchema, FieldSpec
from .segment import Batch, Segment
from .vector import VectorWriter

__version__ = "0.2.0"

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
    "Sampler",
    "Segment",
    "SegmentDataset",
    "SegmentStream",
    "Selection",
    "StorageBackend",
    "UniformSampler",
    "VectorWriter",
    "ZarrBackend",
    "get_backend",
    "register_backend",
]