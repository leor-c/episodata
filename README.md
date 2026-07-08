# Episodata

A unified episode dataset library for world models and control — robotics,
video games, or any sequential-decision domain. One logical data model that
works for toy projects and scales to large multimodal datasets by swapping
the storage backend, never the API.

The design (see `world_model_dataset_design.md`) separates three layers:

1. **Logical data model** — schema, spaces, fields, observations
2. **Query and sampling API** — episodes, segments, transitions
3. **Storage implementation** — a replaceable `StorageBackend`

## Quick start

```python
import numpy as np
from episodata import Dataset

episodes = [{
    "observations": {
        "front_camera": np.zeros((100, 3, 64, 64), dtype=np.uint8),
        "state": np.zeros((100, 7), dtype=np.float32),
    },
    "actions": np.zeros((100, 4), dtype=np.float32),
    "rewards": np.zeros(100, dtype=np.float32),
    "terminated": True,     # separate Gymnasium-style signals;
    "truncated": False,     # bool or per-step array
}]

# In memory (toy) ...
dataset = Dataset.from_episodes(episodes)
# ... or persisted; same API from here on.
dataset = Dataset.from_episodes(episodes, path="my_dataset")
dataset = Dataset.open("my_dataset")
```

### Fields: flat storage, schema-driven structure

Observations, actions and rewards are all *fields* — named arrays grouped
into *spaces* (shared shape/dtype). The hierarchy comes from the schema,
not from nesting in storage. Episode reads return a `Segment` holding every
field plus per-step `terminated`/`truncated`/`mask` flags:

```python
seg = dataset.episode(0).segment(0, 8)
seg["front_camera"]         # flat access
seg.image.front_camera      # space access
seg.image.stacked()         # same-space fields stack safely
for key, value in seg.image.items(): ...

seg.action, seg.reward      # bare action/reward arrays resolve directly
seg.observations            # role views: observation-role fields only
seg.actions, seg.rewards    # ... action / reward roles, always collections
seg.terminated              # [L] flag, True only on a terminal final step
```

### Hierarchical fields (complex actions and observations)

Minecraft-style structured actions — or nested observation dicts — flatten
into stable path keys (`"keyboard/w"`); storage and the storage boundary
stay flat, and the hierarchy is rebuilt at the access layer:

```python
episode = {
    "observations": {"pov": pov, "inventory": {"stone": s, "wood": w}},
    "actions": {"camera": cam, "keyboard": {"w": fwd, "jump": jmp}},
    "rewards": rewards,
}
dataset = Dataset.from_episodes([episode])

obs = dataset.episode(0).read()
obs["keyboard/w"]                  # flat access always works
obs.keyboard.w                     # group access
obs.inventory.items()              # iterate a group

dataset.loader(fields=["pov", "keyboard"])   # a prefix selects the subtree
transitions.actions.keyboard.w               # groups work everywhere
```

Name resolution order for attributes and keys: role view, space, group,
field. A *trivial* space — one whose only field carries the space's own
name, as inference produces for a bare action or reward array — resolves
straight to that field's array (`seg.action`, `seg.reward`), not to a
one-entry view; `seg.space_view("action")` returns the view regardless of
field count.

Singular vs plural: singular names resolve to data, while the plural role
views (`observations`, `actions`, `rewards`, `infos`) are collections by
contract — always a `Fields` sub-view, even when the role holds a single
field. `seg.action` is the action data; `seg.actions` is the set of
action-role fields.

### Schema: automatic, declared, or hybrid

```python
from episodata import DatasetSchema, SpaceSpec, FieldSpec

schema = DatasetSchema.infer(example_episode)      # automatic (shape/dtype reliable)
schema = DatasetSchema(spaces=[...], fields=[...]) # declared
dataset.rename_space("vector", "proprio")          # hybrid: refine + persist
```

The persisted schema is authoritative — it is never re-inferred on reopen.

### Sampling

```python
# Fixed-length segments / context+target segments for world-model training
loader = dataset.loader(
    fields=["front_camera", "state", "action"],
    context_length=4,
    target_length=32,
    batch_size=64,
    seed=0,
)
batch = loader.sample()             # arrays [B, L, ...]
batch.context, batch.target         # time-sliced views
batch.terminated                    # [B, L] done flags

# Sequential scan (evaluation, statistics)
for batch in dataset.loader(sequence_length=32, shuffle=False): ...

# Transitions for control
t = dataset.sample_transitions(batch_size=256)
t.observations, t.actions, t.rewards, t.next_observations, t.terminated

# Filtering
dataset.loader(sequence_length=8, filter=lambda ep: ep.terminated)
```

### Map-style access (`torch.utils.data.DataLoader`)

`loader()` is an infinite, shuffled, with-replacement stream. `segments()`
gives the same fixed-length segments as an indexable, map-style dataset
instead — its `__len__`/`__getitem__` satisfy `DataLoader`'s map-style
protocol by duck typing, so reads (including per-episode decompression on
disk-backed backends) get sharded across `num_workers` worker processes:

```python
from torch.utils.data import DataLoader

segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=8)
loader = DataLoader(
    segments, batch_size=32, shuffle=True,
    num_workers=4, collate_fn=segments.collate,
)
for batch in loader: ...   # episodata.Batch, arrays [B, L, ...]
```

`episodata` itself never imports torch — `segments[i]` returns a `Segment`
(unbatched arrays `[L, ...]`) and works standalone with no torch installed.


### Online episode append

The write API mirrors the Gymnasium loop one-to-one: `add_reset` records
what `env.reset()` returned, then each `add_step` records one `env.step`
call — the action sent plus everything the env returned, including the
separate `terminated` / `truncated` signals. A True signal finalizes the
episode, exactly as it ends the Gymnasium episode:

```python
obs, info = env.reset()
writer = dataset.new_episode()
writer.add_reset(obs, infos=info)            # dummy zero action/reward

while True:
    obs, reward, terminated, truncated, info = env.step(action)
    writer.add_step({
        "observations": obs, "actions": action, "rewards": reward,
        "terminated": terminated, "truncated": truncated, "infos": info,
    })
    if terminated or truncated:
        break                                # episode already finalized

dataset.episode(writer.episode_id).terminated
```

Episodes ended for other reasons (e.g. a collection-time limit) are closed
explicitly with `writer.end(truncated=True)`.

Writers are stateless handles: only the ``episode_id`` needs to be kept.
An episode can be continued later without the original writer — by
reattaching one, or by calling the id-based methods on the dataset
directly:

```python
writer = dataset.resume_episode(episode_id)        # reattach a writer
writer = dataset.episode(episode_id).writer()      # same, via the view

dataset.add_step(episode_id, step)                 # or skip the writer
dataset.add_steps(episode_id, segment)
dataset.end_episode(episode_id, terminated=True)
```

This also works after `Dataset.open` on a persistent backend (ongoing
episodes survive `flush()` / reopen). New episodes become sampleable by
existing loaders immediately.

## Storage backends

Built-in:

- `memory` — in-memory, for toy datasets, tests, replay-buffer usage
- `npz_directory` — one compressed `.npz` per episode plus a `manifest.json`
  holding the logical schema and the storage manifest
- `zarr` — one chunked store for the whole dataset (Zarr v3): each field is
  a single array concatenated along time, plus O(1) per-episode index
  writes. For datasets that outgrow one-file-per-episode (many thousands of
  episodes, or short-segment sampling from long episodes — only the chunks
  overlapping a read are decoded). Optional dependency:
  `pip install "episodata[zarr]"` (needs Python ≥ 3.11)

`Dataset.open(path)` reads the backend name from the manifest, so opening
code never changes when a dataset changes backend. Migrate a dataset that
outgrew its backend by streaming it across the storage boundary:

```python
big = Dataset.open("my_dataset").copy_to("my_dataset_zarr", backend="zarr")
```

A backend implements `StorageBackend` (`episodata/backends/base.py`): reads
of logical field ids over temporal selections, online appends, and the
episode index. Everything physical — layout, shards, codecs, chunking,
caching, decoding — is the backend's concern; it must expose the *logical*
representation declared by the schema regardless of physical encoding.
Backends may return field-oriented dicts or space-oriented `SpaceBlock`s;
the high-level layer normalizes both.

```python
from episodata import StorageBackend, register_backend

@register_backend
class MyBackend(StorageBackend):
    name = "my_backend"
    ...
```

## Conventions (v1)

- All temporal fields of an episode share one length `T`, aligned
  **action-in**: row `t` holds the action and reward that *led to*
  observation `t`. Row 0 is the reset row — the initial observation with
  dummy zero action/reward (`writer.add_reset(obs)`).
- A transition is `(obs[t], action[t+1], reward[t+1], obs[t+1], done[t+1])`;
  `sample_transitions` does this pairing, so its `(s, a, r, s', done)`
  output is convention-free.
- `terminated` and `truncated` are separate signals, as in Gymnasium. The
  write API accepts them per step (a True value finalizes the episode, and
  is only legal on the final step); storage keeps them as episode-level
  flags, and per-step flags in batches are derived (`True` only on an
  episode's final step — which under action-in is exactly the row whose
  `env.step` reported the signal).
- Field keys are the stable logical identifiers used across the storage
  boundary.

### Action-out data

Pipelines that pair each observation with the action taken *at* it
(D4RL-style) convert at the write boundary; storage stays canonical and
everything downstream is shared:

```python
from episodata import ActionOutWriter

# bulk import: rows are shifted at write time
dataset = Dataset.from_episodes(episodes, alignment="action_out")

# online collection: obs written immediately, action/reward held one step
writer = ActionOutWriter(dataset.new_episode())
writer.add_step({"observations": o, "actions": a, "rewards": r})
writer.end(terminated=True, final_observation=last_obs)
```

Note the last action/reward of an action-out episode pair with an
observation that was never recorded; pass `final_observation` (or accept
that they are dropped — no transition could use them anyway).

## Development

```bash
uv pip install -e ".[dev]"
pytest tests
```

The test suite runs against every registered backend to enforce the
storage-boundary contract.
