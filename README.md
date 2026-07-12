# Episodata

A unified episode dataset library for world models and control — robotics,
video games, or any sequential-decision domain. One logical data model that
works for toy projects and scales to large multimodal datasets by swapping
the storage backend, never the API.

The design separates three layers:

1. **Logical data model** — schema, fields, observations
2. **Query and sampling API** — episodes, segments, transitions
3. **Storage implementation** — a replaceable `StorageBackend`

New here? [`docs/getting_started.md`](docs/getting_started.md) covers the
design and basic usage in a few minutes, and
[`examples/getting_started.ipynb`](examples/getting_started.ipynb) is a
runnable tour. This README is the full reference;
[`docs/api.md`](docs/api.md) lists the module layering and key signatures
at a glance.

## Quick start

```python
import numpy as np
from episodata import Dataset

episodes = [{
    "initial_observation": {  # what env.reset() returned
        "front_camera": np.zeros((3, 64, 64), dtype=np.uint8),
        "state": np.zeros(7, dtype=np.float32),
    },
    "observations": {         # one entry per step, like every other field
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

### Fields: flat storage, role-first access

Observations, actions and rewards are all *fields* — flat named arrays;
structure is rebuilt at the access layer. Every read returns a `Segment` of
*transitions*: each entry pairs the observation an action was taken at with
that action, its reward, and the observation it produced — the names say
what pairs with what, so there is no alignment convention to learn.

Access is strictly hierarchical: role first, then field. A role holding one
bare array (a non-dict source, as a plain Box action or observation
produces) resolves straight to that array; anything dict-shaped is a view:

```python
seg = dataset.episode(0).segment(0, 8)   # transitions [0, 8)
seg.observation.front_camera   # [8, ...] the obs each action was taken at
seg.next_observation.front_camera  # ... and the obs each action produced
seg.action, seg.reward         # [8, ...] bare arrays resolve directly
seg.terminated                 # [8] done flag of each transition

for key, value in seg.obs.items(): ...
```

Singular, plural and the `obs` shorthand are aliases for the same object:
`seg.observation` == `seg.obs` == `seg.observations`, and likewise
`seg.next_obs`, `seg.actions`, `seg.rewards`, `seg.infos`. There are no
other shortcuts — fields are reached only through their role.

`seg.obs[k]`, `seg.next_obs[k]` and the action/reward arrays are zero-copy
views into one shared row buffer — pixel observations are never duplicated.
Lengths always count env steps: an episode that took `T` `env.step` calls
has `episode.length == T` and reads as `T` transitions, with the reset
observation surfacing as `seg.obs[k][0]` of a segment starting at 0.

### Hierarchical fields (complex actions and observations)

Minecraft-style structured actions — or nested observation dicts — flatten
into stable path keys (`"keyboard/w"`); storage and the storage boundary
stay flat, and the hierarchy is rebuilt at the access layer:

```python
episode = {
    "initial_observation": {"pov": pov0, "inventory": {"stone": s0, "wood": w0}},
    "observations": {"pov": pov, "inventory": {"stone": s, "wood": w}},
    "actions": {"camera": cam, "keyboard": {"w": fwd, "jump": jmp}},
    "rewards": rewards,
}
dataset = Dataset.from_episodes([episode])

seg = dataset.episode(0).read()
seg.action["keyboard/w"]           # flat path keys always work under a role
seg.action.keyboard.w              # group access
seg.obs.inventory.items()          # iterate a group

dataset.segment_stream(fields=["pov", "keyboard"])   # a prefix selects the subtree
transitions.action.keyboard.w                # groups work everywhere
```

Within a role view a name is an exact field or a group prefix — nothing
else. Method names (`schema`, `keys`/`items`/`values`/`get`) win attribute
lookup over a same-named field; brackets always reach the field.

### Schema: automatic or declared

Every field carries its own per-step format — shape, dtype and optional
bounds/layout — the same per-leaf model as a Gymnasium `Dict` space:

```python
from episodata import DatasetSchema, FieldSpec
from episodata.utils import schema_from_gym_spaces

schema = DatasetSchema.infer(example_episode)      # automatic (shape/dtype reliable)
schema = schema_from_gym_spaces(env.observation_space, env.action_space)  # from a Gymnasium env
schema = DatasetSchema(fields=[
    FieldSpec("front_camera", shape=(3, 64, 64), dtype="uint8", low=0, high=255),
    FieldSpec("action", shape=(4,), dtype="float32", role="action"),
])                                                 # declared
```

The persisted schema is authoritative — it is never re-inferred on reopen.

### Sampling

All lengths count transitions (env steps):

```python
# Fixed-length segments / context+target segments for world-model training
stream = dataset.segment_stream(
    fields=["front_camera", "state", "action"],
    context_length=4,
    target_length=32,
    batch_size=64,
    seed=0,
)
batch = stream.sample()             # arrays [B, L, ...]
batch.context, batch.target         # time-sliced views; target.observation
                                    # starts where context.next_observation ends
batch.terminated                    # [B, L] done flags

# Sequential scan (evaluation, statistics)
for batch in dataset.segment_stream(sequence_length=32, shuffle=False): ...

# Transitions for control: a time-squeezed batch, arrays [B, ...]
t = dataset.sample_transitions(batch_size=256)
t.obs, t.action, t.reward, t.next_obs, t.terminated

# Filtering
dataset.segment_stream(sequence_length=8, filter=lambda ep: ep.terminated)
```

### Map-style access (`torch.utils.data.DataLoader`)

`segment_stream()` is an infinite, shuffled, with-replacement stream. `segments()`
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

### Converting to tensors (`Segment.map`)

Within a segment, `observation` and `next_observation` are two views of one
row buffer. A copying conversion applied per accessor — a device transfer,
`pin_memory`, anything forcing contiguity — would materialize each view
separately and duplicate the overlapping rows. `map(fn)` converts in one
pass instead: `fn` runs once per field's underlying buffer, and every
accessor of the result is re-derived as a view of what `fn` returned, so
the sharing survives the conversion:

```python
batch = batch.map(lambda a: torch.as_tensor(a).to("cuda"))

batch.obs.front_camera       # cuda tensor ...
batch.next_obs.front_camera  # ... two views of one allocation
```

The result is a regular `Batch` — role-first access, `context` / `target`
and the per-transition flags all keep working. `fn` may return any
array-like supporting basic slicing; episodata never imports the target
framework.

When the sequence itself is wanted (sequence models, video), a window's
`L + 1` underlying observations are exposed directly:
`seg.all_observations` (alias `all_obs`; likewise `all_infos`) — `[:-1]`
is `observation`, `[1:]` is `next_observation`.

### Starting a new dataset from scratch

The most common starting point isn't a pile of arrays — it's an empty
dataset and a live env. `Dataset.create` takes a schema and no episodes,
giving a zero-row dataset that's immediately writable:

```python
import gymnasium as gym
from episodata import Dataset
from episodata.utils import schema_from_gym_spaces

env = gym.make("CartPole-v1")
schema = schema_from_gym_spaces(env.observation_space, env.action_space)
dataset = Dataset.create(schema, path="cartpole_data")  # drop path to keep it in memory

for _ in range(10):
    obs, info = env.reset()
    writer = dataset.new_episode(obs, infos=info)
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        writer.add_step({
            "observations": obs, "actions": action, "rewards": reward,
            "terminated": terminated, "truncated": truncated,
        })

dataset.num_episodes  # 10, sampleable immediately — see Sampling above
```

### Online episode append

The write API mirrors the Gymnasium loop one-to-one: an episode begins at
reset, so `new_episode` takes what `env.reset()` returned — the initial
observation. Each `add_step` then records one `env.step` call — the action
sent plus everything the env returned, including the separate
`terminated` / `truncated` signals. A True signal finalizes the episode,
exactly as it ends the Gymnasium episode:

```python
obs, info = env.reset()
writer = dataset.new_episode(obs, infos=info)

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

### Vectorized environments

For N parallel envs, `dataset.vector_writer()` keeps one ongoing episode
per env and handles their staggered boundaries with next-step autoreset
semantics (the Gymnasium 1.0 vector default): a done env's next observation
starts a fresh episode as its initial observation. Plain arrays in, no env-library
imports — any vec env source works:

```python
vec = dataset.vector_writer()
obs, infos = envs.reset(seed=0)
vec.reset(obs)

for _ in range(num_steps):
    obs, rewards, terminated, truncated, infos = envs.step(actions)
    vec.step(obs, actions=actions, rewards=rewards,
             terminated=terminated, truncated=truncated)

vec.close()  # still-ongoing episodes are finalized as truncated
```

See [the getting-started guide](docs/getting_started.md#collecting-from-vectorized-environments)
for details, including the recipe for same-step-autoreset envs (older
Gymnasium, SB3), which drive one writer per env instead.

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

```python
from episodata import StorageBackend, register_backend

@register_backend
class MyBackend(StorageBackend):
    name = "my_backend"
    ...
```

## Conventions (v1)

The user-facing contract has **no alignment convention to learn** — both
boundaries speak env steps, in Gymnasium's own vocabulary:

- **Writes**: every temporal field carries one entry per step. Online,
  `new_episode` records the reset observation and each `add_step` one
  `env.step`. In bulk, the episode dict is **self-describing** through its
  boundary key: `initial_observation` (the reset observation) marks
  action-in — `actions[t]` *led to* `observations[t]` — while
  `final_observation` marks action-out; carrying both is an error. The
  `alignment` argument is only needed for action-out data without its
  final observation, which carries no key.
- **Reads**: every read is a window of *transitions* with explicitly named,
  transition-aligned arrays — `observations` (where each action was taken),
  `actions`/`rewards`, `next_observations` (what each action produced), and
  per-transition `terminated`/`truncated`/`mask` flags. An episode of `T`
  steps has `episode.length == T` and exactly `T` transitions.
- `infos` pair with observations (Gymnasium's `info` accompanies both
  `reset()` and `step()`) and are all-or-nothing: supplying `infos` in bulk
  requires the matching `initial_info` for the reset observation, and vice
  versa — an arbitrary info dict has no universal zero sentinel, so it is
  never zero-filled. Both may be omitted entirely.
- `terminated` and `truncated` are separate signals, as in Gymnasium. The
  write API accepts them per step (a True value finalizes the episode, and
  is only legal on the final step); storage keeps them as episode-level
  flags, and the per-transition flags in reads are derived (`True` only on
  an episode's final transition — the `env.step` that reported the signal).
- Field keys are the stable logical identifiers used across the storage
  boundary.

**Internal storage layout** (relevant only to backend implementers): a
`T`-step episode is stored as `T + 1` equal-length rows in action-in
alignment — row 0 is the reset row (initial observation, zero-filled
action/reward), and row `t` holds the action and reward that led to
observation `t`. The query layer reads `L + 1` rows per `L`-transition
window and never exposes row indices or the zero-filled slots.

### Action-out data

Pipelines that pair each observation with the action taken *at* it
(D4RL-style) convert at the write boundary; storage stays canonical and
every read is shared — the transition view hands the pairing back exactly
as the source meant it (`actions[i]` taken at `observations[i]`):

```python
from episodata import ActionOutWriter

# bulk import: equal-length observations/actions/rewards, actions[t] taken
# AT observations[t]. The final_observation key alone marks the episode as
# action-out — no alignment argument needed
episodes = [{
    "observations": obs, "actions": acts, "rewards": rews,
    "final_observation": last_obs,   # mirrors ActionOutWriter.end
    "terminated": True,
}]
dataset = Dataset.from_episodes(episodes)

# online collection: obs written immediately, action/reward held one step
writer = ActionOutWriter(dataset.new_episode())
writer.add_step({"observations": o, "actions": a, "rewards": r})
writer.end(terminated=True, final_observation=last_obs)
```

The last action/reward of an action-out episode pair with an observation
that was never recorded; pass `final_observation` (and `final_info`, if
using infos) to keep them, or accept that they are dropped — no transition
could use them anyway. Data *without* a final observation carries no
boundary key, so that one case states its alignment explicitly:

```python
dataset = Dataset.from_episodes(episodes, alignment="action_out")
```

## Development

```bash
uv pip install -e ".[dev]"
pytest tests
```

The test suite runs against every registered backend to enforce the
storage-boundary contract.
