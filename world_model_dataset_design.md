# Unified World Model Dataset Design

## Goal

Provide a simple, use-case-driven dataset API that works for toy projects and can scale to very large multimodal datasets without changing the high-level data model.

The design separates:

1. **Logical data model**
2. **Query and sampling API**
3. **Storage implementation**

The logical API should remain stable even when the storage backend is replaced.

---

## Core Data Model

### Dataset

A dataset contains episodes and a persistent schema.

```python
dataset.schema
dataset.num_episodes
dataset.episode(episode_id)
```

### Episode

An episode is a logical trajectory, not a physical file. It contains temporal fields such as observations, actions, rewards, termination, and truncation signals.
Optionally, also info (as in Gym interface). 

### Fields

Observations, actions, rewards and infos are structurally the same thing: schema-backed collections of named fields. One generic container (`Fields`) serves all of them; the *role* of a field (observation / action / reward / info) is schema metadata, not a separate container type.

Fields are grouped into **spaces**:

```python
fields.image.front
fields.image.wrist
fields.proprio.state
```

Equivalent dictionary-style access should always be available:

```python
fields["front"]
fields.image["front"]
```

Iteration over a space is supported:

```python
for key, value in fields.image.items():
    ...
```

Role-based sub-views select by schema role:

```python
fields.observations    # observation-role fields only
fields.actions
fields.rewards
```

The internal representation may remain flat. The hierarchy is reconstructed from the schema.

### Segment

Episode reads return a **segment**: all temporal fields of a contiguous run of steps — observations, actions, rewards, infos — plus per-step episode-boundary metadata (`terminated` / `truncated` flags and a padding `mask`) aligned with the leading dims. A batch is a segment with an extra leading batch dimension.



---

## Spaces and Fields

### SpaceSpec

A space describes the shared logical format of a set of fields.

```python
SpaceSpec(
    key="image",
    shape=(3, 256, 256),
    dtype="uint8",
    low=0,
    high=255,
    layout="CHW",
)
```

The user may explicitly provide the space key. Automatically inferred spaces may initially receive generated keys that can later be renamed.

Fields in the same space share the guarantees required for common structural operations such as stacking, validation, and generic transforms.

### FieldSpec

Each field has a stable identity and belongs to one space.

```python
FieldSpec(
    key="front_camera",
    space="image",
    semantic_type="rgb",
)
```

Field-level metadata may include:

- bounds
- units
- semantic type
- optionality
- temporal alignment
- missing-value behavior

Space membership and semantic meaning remain separate concepts.

Two spaces may have identical format specifications but different user-defined keys.

---

## Schema Construction

The library should support three modes.

### Automatic

Infer structural spaces from example data.

Reliable inference includes:

- shape
- dtype

Properties such as bounds, semantics, categorical meaning, and layout may remain unknown.

### Declared

The user defines the complete schema explicitly.

### Hybrid

Infer the initial schema and then rename, refine, or override selected spaces and fields.

The persisted schema becomes authoritative after dataset creation. It should not be re-inferred whenever the dataset is reopened.

---

## Query API

The API should express semantic requests rather than storage operations.

Core operations include:

- full episode access
- transition sampling
- fixed-length segment sampling
- context and target segments
- online episode append (create new episode (potentially with initial step or segment), add step / segment to existing episode, add complete episode)
- query whether an episode has terminated or not 
- recurrent burn-in
- arbitrary temporal offsets
- field selection
- filtering
- sequential scans
- task-balanced sampling
- prioritized sampling

Example:

```python
stream = dataset.segment_stream(
    fields=["front_camera", "state", "action"],
    context_length=4,
    target_length=32,
    batch_size=64,
)
```

The query may be represented declaratively so that the backend can optimize execution, batch nearby reads, push down field selection, and avoid unnecessary decoding.

---

## Storage Boundary

The storage backend should operate on stable logical field identifiers and temporal selections.

```python
backend.read_fields(
    field_ids=["front_camera", "state"],
    selection=segment_selection,
)
```

The backend is responsible for:

- physical field locations
- files and shards
- codecs
- chunking
- indexing
- reading and writing
- caching
- prefetching
- decoding
- converting physical representations into logical ones

The high-level layer is responsible for:

- schema interpretation
- observation structure
- field-to-space grouping
- user-facing queries
- sampling policies
- batch construction

The storage backend does not need to implement Python access such as:

```python
obs.image.front
```

---

## Logical and Physical Representations

The logical representation may differ from the stored representation.

Example:

```python
SpaceSpec(
    key="image",
    shape=(3, 256, 256),
    dtype="uint8",
    layout="CHW",
)
```

The backend may physically store the same field as H.264 video decoded into HWC frames. It must expose the logical CHW representation through the stable API.

This distinction allows different storage backends to preserve the same dataset semantics.

---

## Persisted Metadata

The dataset should persist both logical and physical metadata.

### Logical schema

- schema version
- space keys and specifications
- field keys and stable IDs
- field-to-space assignments
- semantic metadata
- temporal metadata
- optionality and validity rules

### Storage manifest

- backend type
- physical locations
- shard mappings
- codecs
- physical dtype and layout
- chunk sizes
- indexes and offsets
- checksums

Conceptually:

```python
DatasetManifest(
    schema=DatasetSchema(...),
    storage=StorageManifest(...),
)
```

The two sections may be serialized together, but they should remain conceptually separate.

---

## Backend Flexibility

A backend may return data in different physical organizations.

### Field-oriented

```python
{
    "front_camera": tensor_a,
    "wrist_camera": tensor_b,
}
```

### Space-oriented

```python
SpaceBlock(
    space="image",
    keys=("front_camera", "wrist_camera"),
    data=stacked_tensor,
)
```

The high-level API should normalize both forms into the same observation interface.

This allows efficient contiguous or stacked representations without forcing all backends to use them.

---

## Efficiency Principles

The abstraction should avoid forcing:

- scalar-only access
- complete episode materialization
- eager loading of all modalities
- a single physical layout
- a fixed execution order
- nested physical storage matching the logical hierarchy

The API should fix semantics while leaving execution flexible.

Optional capability discovery, cost hints, backend-specific execution options, and a low-level native escape hatch may be provided for unusual workloads.

---

## Simple and Scalable Usage

Toy usage:

```python
dataset = Dataset.from_episodes(episodes)

stream = dataset.segment_stream(
    sequence_length=32,
    batch_size=64,
)
```

Large-scale usage:

```python
dataset = Dataset.open("s3://bucket/data")

stream = dataset.segment_stream(
    sequence_length=32,
    batch_size=64,
    workers=32,
    cache="local_nvme",
)
```

Both should use the same logical concepts and observation API.

---

## Central Principle

The design should preserve:

\[
\boxed{
\text{stable semantics}
+
\text{use-case-driven queries}
+
\text{replaceable storage}
}
\]

The schema defines what the data means. The query API defines what the user wants. The backend decides how to execute it efficiently.

---

## Implemented API (v1)

The implementation layers modules so that each depends only on the ones above it:

| Module | Layer | Contents |
|---|---|---|
| `schema.py` | logical spec | `SpaceSpec`, `FieldSpec`, `DatasetSchema` — serializable, authoritative, no array data |
| `fields.py` | generic field views | `Fields` (flat named arrays + space/group/role access), `SpaceView`, `FieldGroup` |
| `segment.py` | temporal containers | `Segment` (fields + per-step flags), `Batch` (leading batch dim, context/target slicing) |
| `episode.py` | trajectory views | `Episode` (lazy read view), `EpisodeWriter` (online append handle) |
| `sampling.py` | query & sampling | `SegmentStream` (stream), `SegmentDataset` (map-style), `TransitionBatch`, `SegmentIndex` |
| `dataset.py` | entry point | `Dataset` — ties schema, backend, episodes and queries together |
| `backends/` | storage | `StorageBackend` contract; `memory`, `npz_directory` |
| `normalize.py` | write boundary | canonical episode/step dicts, `/`-path flattening, alignment shifts |
| `action_out.py` | write boundary | `ActionOutWriter` — D4RL-style alignment converted at write time |

The spec layer (`schema.py`) never touches array data; the field views (`fields.py`) are runtime views that depend on the schema, not the other way around. Storage sees only flat field keys and temporal selections — spaces, groups, roles and segments are all reconstructed above the storage boundary.

### Key signatures

**Dataset** — construction, writes, queries:

```python
class Dataset:
    @classmethod
    def create(cls, schema, path=None, backend=None, **backend_options) -> Dataset
    @classmethod
    def from_episodes(cls, episodes, schema=None, path=None, backend=None,
                      alignment="action_in", **backend_options) -> Dataset
    @classmethod
    def open(cls, path, backend="npz_directory") -> Dataset

    schema: DatasetSchema
    num_episodes: int
    def episode(self, episode_id: int) -> Episode
    def episodes(self) -> Iterator[Episode]

    def add_episode(self, episode, alignment="action_in") -> Episode
    def new_episode(self, initial=None) -> EpisodeWriter
    def resume_episode(self, episode_id: int) -> EpisodeWriter
    def add_reset(self, episode_id, observations, infos=None) -> None
    def add_step(self, episode_id, step) -> None
    def add_steps(self, episode_id, steps) -> None
    def end_episode(self, episode_id, terminated=False, truncated=False) -> Episode

    def segment_stream(self, fields=None, batch_size=1, sequence_length=None,
               context_length=None, target_length=None, shuffle=True,
               seed=None, filter=None, pad="suffix") -> SegmentStream
    def segments(self, fields=None, sequence_length=None, context_length=None,
                 target_length=None, filter=None, pad="suffix") -> SegmentDataset
    def sample_transitions(self, batch_size, fields=None, seed=None,
                           filter=None) -> TransitionBatch

    def rename_space(self, old: str, new: str) -> None
```

**Episode / EpisodeWriter** — lazy reads and online appends:

```python
class Episode:
    id: int
    length: int
    terminated: bool          # episode-level flags
    truncated: bool
    ongoing: bool
    def segment(self, start=0, stop=None, fields=None) -> Segment   # arrays [L, ...]
    def step(self, t: int, fields=None) -> Segment                  # no leading time dim
    def read(self, fields=None) -> Segment                          # full episode
    def writer(self) -> EpisodeWriter

class EpisodeWriter:            # stateless handle, keyed by episode_id
    def add_reset(self, observations, infos=None) -> None
    def add_step(self, step) -> None       # True terminated/truncated finalizes
    def add_steps(self, steps) -> None     # leading time dim
    def end(self, terminated=False, truncated=False) -> Episode
```

**Fields / Segment / Batch** — the container hierarchy:

```python
class Fields(Mapping):
    fields["front_camera"]        # flat field access (also "keyboard/w" paths)
    fields.image.front_camera     # space attribute access -> SpaceView
    fields.keyboard.w             # group attribute access -> FieldGroup
    observations: Fields          # role views (schema-driven)
    actions: Fields
    rewards: Fields
    infos: Fields
    schema: DatasetSchema
    def select(self, fields: list[str]) -> Fields

class Segment(Fields):            # arrays [L, ...] (or unbatched single step)
    terminated: np.ndarray        # True only on a terminal final step
    truncated: np.ndarray
    mask: np.ndarray              # True on real steps, False on padding

class Batch(Segment):             # arrays [B, L, ...], flags [B, L]
    context: Batch                # time slices when configured with
    target: Batch                 # context_length / target_length
```

**SegmentStream / SegmentDataset** — SegmentStream is a thin sampling policy over
SegmentDataset (its `segments` attribute), which owns all segment
reading, padding and collation:

```python
class SegmentDataset:             # map-style; torch DataLoader-compatible
    def __len__(self) -> int
    def __getitem__(self, i: int) -> Segment
    def collate(self, items: list[Segment]) -> Batch   # pass as collate_fn
    def refresh(self) -> None     # re-snapshot index; no-op unless data changed

class SegmentStream:              # infinite shuffled stream / sequential scan
    segments: SegmentDataset      # refreshed on every sample()
    def sample(self) -> Batch
    def sample_transitions(self) -> TransitionBatch
    def __iter__(self) -> Iterator[Batch]

@dataclass
class TransitionBatch:            # alignment-free (s, a, r, s', done)
    observations: Fields
    actions: Fields
    rewards: np.ndarray | None
    next_observations: Fields
    terminated: np.ndarray
    truncated: np.ndarray
```

**Schema** — the persistent logical spec:

```python
SpaceSpec(key, shape, dtype, low=None, high=None, layout=None, metadata={})
FieldSpec(key, space, role="observation", semantic_type=None, optional=False, metadata={})

class DatasetSchema:
    def __init__(self, spaces: Iterable[SpaceSpec], fields: Iterable[FieldSpec])
    @classmethod
    def infer(cls, example_episode) -> DatasetSchema
    def field_keys(self, role=None) -> list[str]
    def fields_in_space(self, space_key) -> list[str]
    def rename_space(self, old, new) -> None
    def to_dict() / from_dict() / to_json() / from_json()
```

**StorageBackend** — the storage boundary (flat field ids + temporal selections):

```python
Selection(episode_id, start, stop)

class StorageBackend(ABC):
    @classmethod
    def create(cls, schema, path=None, **options) -> StorageBackend
    @classmethod
    def open(cls, path) -> StorageBackend

    schema: DatasetSchema
    num_episodes: int
    def write_schema(self, schema) -> None
    def read_fields(self, field_ids, selection) -> Payload   # field dicts or SpaceBlocks
    def create_episode(self) -> int
    def append_steps(self, episode_id, fields) -> None
    def finalize_episode(self, episode_id, terminated, truncated) -> None
    def episode_length / episode_terminated / episode_truncated / episode_ongoing
```
