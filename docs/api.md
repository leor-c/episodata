# API reference

This is the compact reference for Episodata's public API. Import core types
from `episodata`; framework helpers live in `episodata.utils`.

## Conventions

- All user-facing lengths count transitions (environment steps).
- A segment of length `L` has role arrays shaped `[L, ...]` and an underlying
  observation sequence shaped `[L + 1, ...]`.
- Batched equivalents add a leading `B` dimension.
- Field `shape` is always per step; it excludes time and batch dimensions.
- `terminated` and `truncated` remain separate.
- `mask` is true for real transitions and false for padding.

## Dataset

`Dataset` is the entry point for construction, access, writes, sampling, and
backend migration.

```text
class Dataset:
    @classmethod
    def create(
        cls, schema, path=None, backend=None, **backend_options
    ) -> Dataset

    @classmethod
    def from_episodes(
        cls, episodes, schema=None, path=None, backend=None,
        alignment=None, **backend_options
    ) -> Dataset

    @classmethod
    def open(cls, path, backend=None) -> Dataset

    schema: DatasetSchema
    num_episodes: int
    backend: StorageBackend

    def episode(self, episode_id: int) -> Episode
    def episodes(self) -> Iterator[Episode]
    def __len__(self) -> int

    def add_episode(self, episode, alignment=None) -> Episode
    def new_episode(self, observations=None, infos=None) -> EpisodeWriter
    def resume_episode(self, episode_id: int) -> EpisodeWriter
    def vector_writer(self, num_envs=None) -> VectorWriter
    def add_step(self, episode_id, step) -> None
    def add_steps(self, episode_id, steps) -> None
    def end_episode(
        self, episode_id, terminated=False, truncated=False
    ) -> Episode

    def segments(
        self, fields=None, sequence_length=None,
        context_length=None, target_length=None,
        filter=None, pad="suffix",
    ) -> SegmentDataset

    def segment_stream(
        self, fields=None, batch_size=1, sequence_length=None,
        context_length=None, target_length=None, shuffle=True,
        seed=None, filter=None, pad="suffix", sampler=None,
        read_chunk_size=None,
    ) -> SegmentStream

    def sample_transitions(
        self, batch_size, fields=None, seed=None, filter=None
    ) -> Batch

    def copy_to(self, path=None, backend=None, **backend_options) -> Dataset
    def flush(self) -> None
    def close(self) -> None
```

Backend defaults:

- no `path`: `memory`
- `path` supplied: `zarr`
- `backend="npz_directory"`: one compressed file per episode

`from_episodes()` infers a schema from the first episode when `schema` is
omitted. An `initial_observation` boundary marks action-in input; a
`final_observation` boundary marks action-out input. Pass `alignment` only for
ambiguous input without a boundary key. Persisted schemas are never re-inferred
by `open()`. Bulk `infos` must be paired with `initial_info` (action-in) or
`final_info` (action-out); info rows are never synthesized.

## Schema

```text
FieldSpec(
    key: str,
    shape: tuple[int, ...],
    dtype: str,
    role: str = "observation",
    low: float | None = None,
    high: float | None = None,
    layout: str | None = None,
    optional: bool = False,
)
```

Roles are `observation`, `action`, `reward`, and `info`. `key` is the stable
logical identifier used at the storage boundary. Nested sources become
`/`-separated keys.

```text
class DatasetSchema:
    def __init__(self, fields: Iterable[FieldSpec])

    fields: dict[str, FieldSpec]

    def field(self, key: str) -> FieldSpec
    def field_keys(self, role=None) -> list[str]
    def resolve_fields(self, fields) -> list[str]
    def validate_fields(self, fields) -> dict[str, np.ndarray]

    @classmethod
    def infer(cls, example_episode, alignment=None) -> DatasetSchema

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, value) -> DatasetSchema
    def to_json(self) -> str
    @classmethod
    def from_json(cls, value: str) -> DatasetSchema
```

`resolve_fields()` accepts exact keys and group prefixes. `validate_fields()`
requires a leading time dimension, validates per-step shapes, and casts to the
declared dtype without copying when possible.

Optional Gymnasium conversion:

```text
from episodata.utils import schema_from_gym_spaces

schema_from_gym_spaces(observation_space, action_space) -> DatasetSchema
```

It supports `Box`, `Discrete`, `MultiDiscrete`, `MultiBinary`, `Dict`, and
`Tuple`, and adds a scalar float32 reward field. Requires `episodata[gym]`.

## Episodes and writers

```text
class Episode:
    id: int
    length: int
    terminated: bool
    truncated: bool
    ongoing: bool

    def __len__(self) -> int
    def segment(self, start=0, stop=None, fields=None) -> Segment
    def step(self, t: int, fields=None) -> Segment
    def read(self, fields=None) -> Segment
    def writer(self) -> EpisodeWriter
```

`segment(start, stop)` reads transitions in the half-open interval
`[start, stop)`. Negative indices are supported. `step()` removes the leading
time dimension.

```text
class EpisodeWriter:
    episode_id: int

    def add_step(self, step) -> None
    def add_steps(self, steps) -> None
    def end(self, terminated=False, truncated=False) -> Episode
```

`add_step()` accepts leaves without a time dimension; `add_steps()` accepts
leaves with a leading time dimension. A true final `terminated` or `truncated`
value finalizes the episode. `EpisodeWriter` is a context manager and ends an
open episode with neither flag on clean exit.

```text
class VectorWriter:
    num_envs: int | None
    episode_ids: tuple[int, ...]

    def reset(self, observations, infos=None) -> None
    def step(
        self, observations, actions=None, rewards=None,
        terminated=None, truncated=None, infos=None,
    ) -> None
    def close(self, truncate=True) -> None
```

All inputs carry a leading `num_envs` dimension. `VectorWriter` implements
next-step autoreset semantics and is stateful; it cannot itself be resumed.

```text
class ActionOutWriter:
    def __init__(self, writer: EpisodeWriter)
    episode_id: int
    def add_step(self, step) -> None
    def end(
        self, terminated=False, truncated=False,
        final_observation=None, final_info=None,
    ) -> Episode
```

This adapter converts `(observation_t, action_t, reward_t)` streams into the
canonical write layout. Without `final_observation`, its final pending action
and reward are dropped because their resulting observation is unknown.

## Segments, batches, and fields

```text
class Segment:
    schema: DatasetSchema

    observation                 # aliases: obs, observations
    action                      # alias: actions
    reward                      # alias: rewards
    info                        # alias: infos
    next_observation            # aliases: next_obs, next_observations
    next_info                   # alias: next_infos
    all_observations            # alias: all_obs
    all_infos

    terminated
    truncated
    mask

    def select(self, fields: list[str]) -> Segment
    def map(self, fn) -> Segment

class Batch(Segment):
    context: Batch
    target: Batch
    def select(self, fields: list[str]) -> Batch
    def map(self, fn) -> Batch
```

Role access returns a bare array when that role contains one field created from
a non-dict source. Otherwise it returns `Fields`, a read-only `Mapping`:

```python
segment.obs["front_camera"]
segment.obs.front_camera
segment.action.keyboard.jump
segment.action["keyboard/jump"]
```

Mapping method names win attribute lookup; bracket access always addresses a
field. `map(fn)` applies `fn` once to every underlying row buffer and flag
array, then reconstructs all views. `context` and `target` exist when the batch
was configured with `context_length` and `target_length`.

## Sampling

Sequence configuration follows these rules:

- `sequence_length=N` requests one block of `N` transitions.
- `context_length=C, target_length=T` requests `C + T` transitions and enables
  `batch.context` and `batch.target`.
- With no length argument, the segment length is 1.
- `sequence_length` and `target_length` are mutually exclusive.
- `context_length` requires `target_length`.

```text
class SegmentDataset:
    dataset: Dataset
    fields: list[str]
    segment_length: int
    context_length: int | None
    target_length: int | None
    pad: str | None

    def __len__(self) -> int
    def __getitem__(self, index: int) -> Segment
    def __getitems__(self, indices: Sequence[int]) -> list[Segment]
    def fetch(self, indices: Sequence[int]) -> Batch
    def collate(self, items: list[Segment]) -> Batch
    def refresh(self) -> None
```

The index is a snapshot. `refresh()` rebuilds it only if the backend revision
changed; call it between DataLoader epochs, never during one. `__getitems__()`
is PyTorch DataLoader's batched-fetch hook. `fetch()` skips per-item Segment
objects and returns a `Batch` directly.

```text
class SegmentStream:
    segments: SegmentDataset
    dataset: Dataset
    fields: list[str]
    segment_length: int
    batch_size: int
    sampler: Sampler
    read_chunk_size: int

    def sample(self) -> Batch
    def sample_transitions(self) -> Batch
    def __iter__(self) -> Iterator[Batch]
```

When `shuffle=True`, iteration is infinite and samples with replacement. When
`shuffle=False`, iteration makes one sequential pass. The default
`read_chunk_size` is `max(batch_size, 2048)`; each refill fetches the largest
whole number of batches that fit in that chunk. The segment index refreshes on
refill, so newly written episodes may remain invisible while buffered batches
are consumed.

```text
class Sampler(Protocol):
    def sample(self, index: SegmentIndex, batch_size: int) -> np.ndarray

class UniformSampler:
    def __init__(self, seed=None)
    def sample(self, index: SegmentIndex, batch_size: int) -> np.ndarray
```

Sampler output contains flat indices into the supplied segment index.
`UniformSampler` samples uniformly with replacement.

## TensorDict conversion

```text
from episodata.utils import batch_to_tensordict

batch_to_tensordict(
    batch: Segment,
    device=None,
    include_all_observations=False,
    alignment=None,
) -> TensorDict
```

The default returns the roles at their natural length `L`. Set
`include_all_observations=True` to add the `L + 1` observation sequence.

`alignment="action_in"` or `"action_out"` instead returns an `L + 1` sequence
for every entry: observations use their real shared buffer, while action,
reward, and flag arrays receive one placeholder zero row. Alignment cannot be
combined with `include_all_observations` and requires a time dimension.

Pass `device` here to preserve shared observation storage during conversion.
Requires `torch` and `tensordict`.

## Storage backends

Built-ins:

| Name | Implementation |
|---|---|
| `memory` | process-local NumPy arrays |
| `zarr` | chunked Zarr v3 arrays accessed through TensorStore |
| `npz_directory` | one compressed NumPy archive per episode |

The extension boundary is deliberately small:

```text
Selection(episode_id: int, start: int, stop: int)

class StorageBackend(ABC):
    name: str

    @classmethod
    def create(cls, schema, path=None, **options) -> StorageBackend
    @classmethod
    def open(cls, path) -> StorageBackend

    schema: DatasetSchema
    num_episodes: int
    revision: int

    def episode_length(self, episode_id) -> int
    def episode_terminated(self, episode_id) -> bool
    def episode_truncated(self, episode_id) -> bool
    def episode_ongoing(self, episode_id) -> bool

    def read_fields(
        self,
        field_ids: Sequence[str],
        selections: Sequence[Selection],
    ) -> Sequence[Mapping[str, np.ndarray]]

    def create_episode(self) -> int
    def append_steps(self, episode_id, fields) -> None
    def append_steps_batch(self, episode_ids, fields) -> None
    def finalize_episode(
        self, episode_id, terminated, truncated
    ) -> None
    def flush(self) -> None
    def close(self) -> None
```

`read_fields()` is always batched and preserves selection order. Each returned
array has shape `[selection.length, *field.shape]` in the logical dtype and
layout.

Backends operate on internal action-in rows: an episode with `T` transitions
has `T + 1` storage rows, including its reset row. Consequently,
`StorageBackend.episode_length()` reports row count, while public
`Episode.length` reports transition count. Register an implementation with
`@register_backend`; the backend name is persisted in the dataset manifest.

## Module map

| Module | Responsibility |
|---|---|
| `schema.py` | field identity, formats, validation, serialization |
| `dataset.py` | entry point and write/query factories |
| `episode.py` | lazy episode reads and single-episode writes |
| `segment.py`, `fields.py` | transition-aligned containers and role access |
| `sampling.py`, `sampler.py` | segment indexing, batched fetches, sampling policy |
| `vector.py`, `action_out.py` | collection adapters |
| `backends/` | storage contract and built-in implementations |
| `utils.py` | optional Gymnasium and TensorDict interop |
