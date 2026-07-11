# Getting started

`episodata` stores episodic data — robotics rollouts, game trajectories, any
sequential-decision data — behind one API that doesn't change as your
dataset grows from a handful of toy episodes to millions of steps on disk.

This guide covers the design in a few minutes. For copy-pasteable examples,
see [`examples/getting_started.ipynb`](../examples/getting_started.ipynb).
For the full API, see the [README](../README.md).

## Install

```bash
pip install episodata          # memory + npz_directory backends
pip install "episodata[zarr]"  # + the chunked zarr backend (Python >= 3.11)
```

## Design in three layers

```
logical data model  →  query & sampling API  →  storage backend
   (schema)             (episodes, segments)      (replaceable)
```

1. **Logical data model.** A dataset is episodes plus a schema. The schema
   is named *fields* (`front_camera`, `state`, `action`, `reward`, ...),
   each declaring its own per-step shape and dtype — like the keys of a
   Gymnasium `Dict` space. The schema is data, not code: it's inferred
   automatically or declared up front.

2. **Query and sampling API.** You read data through `Episode` and
   `Segment`, and sample it through `segment_stream` /
   `sample_transitions` — the same calls whether the dataset lives in
   memory or spans a terabyte on disk.

3. **Storage backend.** `StorageBackend` is the one interface a backend
   implements: reads, appends, and the episode index. `memory`,
   `npz_directory`, and `zarr` ship built in; `Dataset.open` reads the
   backend out of the dataset's manifest, so calling code never names one.

The payoff: prototype in memory, `Dataset.from_episodes(..., path=...)` to
persist, and `dataset.copy_to(path, backend="zarr")` to scale up — without
touching a single line downstream.

## One idea worth knowing: everything counts env steps

An episode that took `T` `env.step` calls has length `T` everywhere: `T`
entries per field on write, `episode.length == T`, and `T` *transitions* on
read. The reset observation is written as its own explicitly named key
(`initial_observation` — what `env.reset()` returned), and reads pair every
array by name — `actions[i]` is the action taken at `observations[i]`,
`next_observations[i]` is what it produced — so there is no alignment
convention to learn and no dummy values anywhere, in or out.

`terminated` and `truncated` are separate signals, exactly as in Gymnasium.
D4RL-style "action-out" data (action paired with the observation it was
taken *at*) is converted once at the write boundary, marked by a
`final_observation` key instead of `initial_observation` — see the README's
[Action-out data](../README.md#action-out-data) section — so everything
downstream is shared.

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
    "terminated": True,
}]

dataset = Dataset.from_episodes(episodes)  # schema inferred automatically
```

## Reading data

```python
seg = dataset.episode(0).segment(0, 8)   # transitions [0, 8) of episode 0

seg.obs.front_camera       # [8, ...] the obs each action was taken at
seg.next_obs.front_camera  # ... and the obs each action produced
seg.action, seg.reward     # bare action/reward arrays resolve directly
seg.terminated             # [8] done flag of each transition
```

Access is role-first: `seg.observation` / `seg.obs` / `seg.observations`
are aliases for the same object, and a role holding one bare array (a
non-dict source) resolves straight to that array — hence `seg.action`.
`seg.obs`, `seg.next_obs` and the action/reward arrays are zero-copy views
into one shared row buffer, so consecutive-in-time arrays never duplicate
memory; a segment starting at 0 surfaces the reset observation as
`seg.obs[k][0]`.

## Sampling for training

### Map-style: `segments()` + `DataLoader`

These two subsections use `torch` for the `DataLoader` examples (`pip
install torch`) — a demo-only dependency, not one of episodata's own.

`dataset.segments(...)` is an indexable, map-style view over fixed-length
segments — plain `len()` / `[i]`, so it plugs directly into
`torch.utils.data.DataLoader` for `num_workers` read parallelism (each
worker decompresses its own share of episodes independently):

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
and works standalone with no torch installed.

### To the device: `batch.map`

`obs` and `next_obs` are views of one shared buffer; converting them to
tensors one accessor at a time would copy their overlap twice. `map(fn)`
converts a whole batch in one pass — `fn` runs once per field, and the
result is a regular `Batch` whose views still share storage on the other
side:

```python
for batch in loader:
    batch = batch.map(lambda a: torch.as_tensor(a).to("cuda"))
    batch.obs.front_camera       # cuda tensor ...
    batch.next_obs.front_camera  # ... same allocation — nothing duplicated
```

`fn` is any array → array-like callable, so the same one-liner covers jax,
cupy, dtype casts, or pinned-memory staging. See the README's
[Converting to tensors](../README.md#converting-to-tensors-segmentmap)
section for details, including `all_observations` — direct access to the
`L + 1` observations a window spans.

### Custom samplers: prioritized replay

`segments()` is a plain map-style dataset, so *any* `torch.utils.data.Sampler`
works over it — `DataLoader`'s `sampler` argument replaces `shuffle` with
your own draw order. Prioritization itself is entirely outside episodata:
compute priorities however you like (TD-error, recency, ...) and hand
`DataLoader` a `Sampler` that draws indices accordingly:

```python
from torch.utils.data import Sampler

class PrioritizedSampler(Sampler):
    """Draws segment indices with replacement, weighted by external priorities."""

    def __init__(self, priorities, num_samples, seed=None):
        self.priorities = np.asarray(priorities, dtype=np.float64)
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        probs = self.priorities / self.priorities.sum()
        return iter(self.rng.choice(len(self.priorities), size=self.num_samples, p=probs).tolist())

    def __len__(self):
        return self.num_samples

sampler = PrioritizedSampler(priorities, num_samples=len(segments), seed=0)
loader = DataLoader(segments, batch_size=32, sampler=sampler, collate_fn=segments.collate)
```

Recompute `priorities` and rebuild the sampler as often as your algorithm
needs (each step, each epoch, ...) — episodata only supplies the indexable
segments; how they're drawn is entirely up to the caller.

### Streaming: `segment_stream()`

For a simpler infinite, shuffled, single-process stream — no `DataLoader`
needed — with optional `context` / `target` splitting for world-model
training (all lengths count transitions):

```python
stream = dataset.segment_stream(
    fields=["front_camera", "state", "action"],
    context_length=4, target_length=32,   # or sequence_length=N for one block
    batch_size=64, seed=0,
)
batch = stream.sample()          # arrays [B, L, ...]
batch.context, batch.target      # time-sliced views; target.observation
                                 # starts where context.next_observation ends

transitions = dataset.sample_transitions(batch_size=256)  # arrays [B, ...]
transitions.obs, transitions.action, transitions.next_obs
```

## Persisting and scaling up

```python
dataset = Dataset.from_episodes(episodes, path="my_dataset")  # npz_directory
dataset = Dataset.open("my_dataset")                          # reopen anywhere

# Outgrew one-file-per-episode? Stream across the storage boundary:
big = Dataset.open("my_dataset").copy_to("my_dataset_zarr", backend="zarr")
```

## Collecting data online, two ways

The write API mirrors a Gymnasium rollout one-to-one: `new_episode` records
what `env.reset()` returned (the initial observation), then one `add_step`
call per `env.step`. The usual way is a writer, kept for the lifetime of
the rollout:

```python
writer = dataset.new_episode(obs, infos=info)

obs, reward, terminated, truncated, info = env.step(action)
writer.add_step({
    "observations": obs, "actions": action, "rewards": reward,
    "terminated": terminated, "truncated": truncated,
})
# a True signal finalizes the episode, exactly as it ends the Gym episode
```

A writer is a stateless handle — only its `episode_id` needs to be kept. The
same operations exist directly on `Dataset` by id, useful when you'd rather
not carry a writer object around (e.g. across process boundaries), or want
to append a whole segment in one call instead of step by step:

```python
episode_id = dataset.new_episode(obs).episode_id  # writer discarded; only the id is kept

dataset.add_steps(episode_id, {                 # a whole segment, one call
    "observations": obs_segment, "actions": action_segment,
    "rewards": reward_segment, "terminated": terminated_segment,
})
dataset.end_episode(episode_id, terminated=True)  # or let a True signal finalize it
```

## Collecting from vectorized environments

For N parallel environments, `dataset.vector_writer()` drives one ongoing
episode per env and handles their staggered boundaries: when env `i`
reports `terminated`/`truncated`, its episode finalizes, and on the *next*
step that env's observation starts a fresh episode as its initial observation. This
is next-step autoreset — the Gymnasium 1.0 vector default — and the loop is
just the vector rollout, forwarded:

```python
envs = gym.make_vec("CartPole-v1", num_envs=8)  # any vec env; the writer takes plain arrays

vec = dataset.vector_writer()
obs, infos = envs.reset(seed=0)
vec.reset(obs)

for _ in range(num_steps):
    actions = policy(obs)
    obs, rewards, terminated, truncated, infos = envs.step(actions)
    vec.step(obs, actions=actions, rewards=rewards,
             terminated=terminated, truncated=truncated)

vec.close()  # still-ongoing episodes are finalized as truncated
```

All values carry a leading `num_envs` dimension; the writer never imports
an environment library, so any vec env source works (EnvPool, custom sims,
GPU rollouts after `.cpu().numpy()`). Per-step infos are accepted as a dict
of dense `[N, ...]` arrays — scattering Gymnasium's `_key`-masked info
dicts into dense arrays is up to you.

Vec envs using the *same-step* convention (older Gymnasium, SB3: the done
step already returns the next episode's reset obs, and the true final
observation hides in `infos`) don't need `VectorWriter` — drive one writer
per env; interleaved open episodes are fully supported:

```python
writers = [dataset.new_episode(o) for o in reset_obs]      # one per env
# per env i on each step:
writers[i].add_step({"observations": final_obs_i if done_i else obs[i],
                     "actions": actions[i], "rewards": rewards[i],
                     "terminated": terminated[i], "truncated": truncated[i]})
if done_i:
    writers[i] = dataset.new_episode(obs[i])               # obs[i] is already the reset obs
```

## Next steps

- [`examples/getting_started.ipynb`](../examples/getting_started.ipynb) — a
  runnable tour of everything above.
- [README](../README.md) — full API reference: hierarchical fields,
  schema refinement, backend internals, action-out conversion.
