# Implemented API (v1)

The implementation layers modules so that each depends only on the ones
above it:

| Module | Layer | Contents |
|---|---|---|
| `schema.py` | logical spec | `SpaceSpec`, `FieldSpec`, `DatasetSchema` — serializable, authoritative, no array data |
| `fields.py` | generic field views | `Fields` (one role's flat named arrays + field/group/space access), `SpaceView`, `FieldGroup` |
| `segment.py` | temporal containers | `Segment` (transition-aligned fields + per-transition flags), `Batch` (leading batch dim, context/target slicing) |
| `episode.py` | trajectory views | `Episode` (lazy read view), `EpisodeWriter` (online append handle) |
| `vector.py` | trajectory views | `VectorWriter` — N parallel envs, next-step autoreset |
| `sampling.py` | query & sampling | `SegmentStream` (stream), `SegmentDataset` (map-style), `SegmentIndex` |
| `dataset.py` | entry point | `Dataset` — ties schema, backend, episodes and queries together |
| `backends/` | storage | `StorageBackend` contract; `memory`, `npz_directory`, `zarr` |
| `normalize.py` | write boundary | canonical episode/step dicts, `/`-path flattening, alignment resolution and shifts |
| `action_out.py` | write boundary | `ActionOutWriter` — D4RL-style alignment converted at write time |

The spec layer (`schema.py`) never touches array data; the field views
(`fields.py`) are runtime views that depend on the schema, not the other way
around. Storage sees only flat field keys and temporal selections — spaces,
groups, roles and segments are all reconstructed above the storage boundary.

## Key signatures

**Dataset** — construction, writes, queries:

```python
class Dataset:
    @classmethod
    def create(cls, schema, path=None, backend=None, **backend_options) -> Dataset
        # backend defaults to "memory" without a path, "npz_directory" with one
    @classmethod
    def from_episodes(cls, episodes, schema=None, path=None, backend=None,
                      alignment=None, **backend_options) -> Dataset
        # alignment is normally omitted — episode dicts are self-describing
        # via their initial_observation / final_observation boundary key
    @classmethod
    def open(cls, path, backend=None) -> Dataset
        # backend is read from manifest.json; pass it only for storage without one

    schema: DatasetSchema
    num_episodes: int                    # == len(dataset)
    def episode(self, episode_id: int) -> Episode
    def episodes(self) -> Iterator[Episode]

    def add_episode(self, episode, alignment=None) -> Episode
    def new_episode(self, observations=None, infos=None) -> EpisodeWriter
    def resume_episode(self, episode_id: int) -> EpisodeWriter
    def vector_writer(self, num_envs=None) -> VectorWriter
    def add_step(self, episode_id, step) -> None
    def add_steps(self, episode_id, steps) -> None
    def end_episode(self, episode_id, terminated=False, truncated=False) -> Episode

    def segment_stream(self, fields=None, batch_size=1, sequence_length=None,
               context_length=None, target_length=None, shuffle=True,
               seed=None, filter=None, pad="suffix") -> SegmentStream
    def segments(self, fields=None, sequence_length=None, context_length=None,
                 target_length=None, filter=None, pad="suffix") -> SegmentDataset
    def sample_transitions(self, batch_size, fields=None, seed=None,
                           filter=None) -> Batch    # time-squeezed, arrays [B, ...]

    def rename_space(self, old: str, new: str) -> None
    def rename_field(self, old: str, new: str) -> None
    def copy_to(self, path=None, backend=None, **backend_options) -> Dataset
    def flush(self) -> None
    def close(self) -> None
```

**Episode / EpisodeWriter** — lazy reads and online appends (all lengths
count transitions, i.e. env steps):

```python
class Episode:
    id: int
    length: int               # T env steps == T transitions
    terminated: bool          # episode-level flags
    truncated: bool
    ongoing: bool
    def segment(self, start=0, stop=None, fields=None) -> Segment   # arrays [L, ...]
    def step(self, t: int, fields=None) -> Segment                  # no leading time dim
    def read(self, fields=None) -> Segment                          # full episode
    def writer(self) -> EpisodeWriter

class EpisodeWriter:            # stateless handle, keyed by episode_id;
    episode_id: int             # usable as a context manager
    def add_step(self, step) -> None       # True terminated/truncated finalizes
    def add_steps(self, steps) -> None     # leading time dim
    def end(self, terminated=False, truncated=False) -> Episode
```

**VectorWriter** — collection from N parallel environments with staggered
episode boundaries (next-step autoreset, the Gymnasium 1.0 vector default):

```python
class VectorWriter:                       # stateful — finish with close()
    num_envs: int | None                  # inferred from reset() if omitted
    episode_ids: tuple[int, ...]          # most recent episode id per env
    def reset(self, observations, infos=None) -> None    # leading [num_envs] dim
    def step(self, observations, actions=None, rewards=None,
             terminated=None, truncated=None, infos=None) -> None
    def close(self, truncate=True) -> None    # finalize still-ongoing episodes
```

**ActionOutWriter** — online write-boundary conversion for D4RL-style data
(action taken *at* its observation); storage stays canonical action-in:

```python
class ActionOutWriter:
    def __init__(self, writer: EpisodeWriter)
    def add_step(self, step) -> None      # obs written now, action/reward held one step
    def end(self, terminated=False, truncated=False,
            final_observation=None, final_info=None) -> Episode
```

**Fields / Segment / Batch** — the container hierarchy:

```python
class Fields(Mapping):            # one role's fields, handed out by a Segment
    fields["front_camera"]        # flat field access (also "keyboard/w" paths)
    fields.front_camera           # field attribute access
    fields.keyboard.w             # group attribute access -> FieldGroup
    def space(self, key) -> SpaceView   # explicit space access
    schema: DatasetSchema

class Segment:                    # arrays [L, ...] (or unbatched single step)
    observation: np.ndarray | Fields    # aliases: obs, observations; a role
    action: np.ndarray | Fields         # holding one bare array unwraps to it
    reward: np.ndarray | Fields         # aliases: actions, rewards, info(s),
    info: np.ndarray | Fields           # next_obs, next_observation(s), ...
    next_observation: np.ndarray | Fields
    next_info: np.ndarray | Fields
    terminated: np.ndarray        # True only on a terminal final transition
    truncated: np.ndarray
    mask: np.ndarray              # True on real transitions, False on padding
    schema: DatasetSchema
    def select(self, fields: list[str]) -> Segment

class Batch(Segment):             # arrays [B, L, ...], flags [B, L]
    context: Batch                # time slices when configured with
    target: Batch                 # context_length / target_length
```

**SegmentStream / SegmentDataset** — SegmentStream is a thin sampling policy
over SegmentDataset (its `segments` attribute), which owns all segment
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
    def sample_transitions(self) -> Batch   # time-squeezed: t.obs, t.action,
    def __iter__(self) -> Iterator[Batch]   # t.reward, t.next_obs — all [B, ...]
```

**Schema** — the persistent logical spec:

```python
SpaceSpec(key, shape, dtype, role="observation", low=None, high=None, layout=None, metadata={})
FieldSpec(key, space, role="observation", semantic_type=None, optional=False, metadata={})
# a space belongs to one role; each field's role must match its space's role

class DatasetSchema:
    def __init__(self, spaces: Iterable[SpaceSpec], fields: Iterable[FieldSpec])
    @classmethod
    def infer(cls, example_episode, alignment=None) -> DatasetSchema
    def field(self, key) -> FieldSpec
    def space(self, key) -> SpaceSpec
    def space_of(self, field_key) -> SpaceSpec
    def field_keys(self, role=None) -> list[str]
    def fields_in_space(self, space_key) -> list[str]
    def rename_space(self, old, new) -> None
    def rename_field(self, old, new) -> None
    def to_dict() / from_dict() / to_json() / from_json()
```

**StorageBackend** — the storage boundary (flat field ids + temporal
selections; a `T`-step episode is stored as `T + 1` action-in rows, row 0
being the reset row):

```python
Selection(episode_id, start, stop)

@register_backend                # registers cls.name; Dataset looks backends up by name
class StorageBackend(ABC):
    name: str
    @classmethod
    def create(cls, schema, path=None, **options) -> StorageBackend
    @classmethod
    def open(cls, path) -> StorageBackend

    schema: DatasetSchema
    num_episodes: int
    revision: int                 # bumped on every write; drives SegmentDataset.refresh
    def write_schema(self, schema) -> None
    def read_fields(self, field_ids, selection) -> Payload   # field dicts or SpaceBlocks
    def create_episode(self) -> int
    def append_steps(self, episode_id, fields) -> None
    def append_steps_batch(self, episode_ids, fields) -> None   # one row per episode
    def finalize_episode(self, episode_id, terminated, truncated) -> None
    def episode_length / episode_terminated / episode_truncated / episode_ongoing
    def flush(self) -> None
    def close(self) -> None
```
