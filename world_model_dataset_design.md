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

### Observation

An observation is a schema-backed collection of named fields.

Fields are grouped into **spaces**:

```python
obs.image.front
obs.image.wrist
obs.proprio.state
```

Equivalent dictionary-style access should always be available:

```python
obs["front"]
obs.image["front"]
```

Iteration over a space is supported:

```python
for key, value in obs.image.items():
    ...
```

The internal representation may remain flat. The hierarchy is reconstructed from the schema.



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
- context and target windows
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
loader = dataset.loader(
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

loader = dataset.loader(
    sequence_length=32,
    batch_size=64,
)
```

Large-scale usage:

```python
dataset = Dataset.open("s3://bucket/data")

loader = dataset.loader(
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
