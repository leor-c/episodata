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
   groups named *fields* (`front_camera`, `state`, `action`, `reward`, ...)
   into *spaces* — fields that share a shape and dtype, like the two camera
   feeds of a robot. The schema is data, not code: it's inferred
   automatically, refined, or declared up front.

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

## One convention worth knowing: action-in alignment

Every episode has one length `T`. Row `t` holds the action and reward that
*led to* observation `t` — row 0 is the reset row (`env.reset()`'s
observation, with a dummy zero action/reward). This is the natural shape of
a Gymnasium rollout, and it makes `sample_transitions` unambiguous:
`(obs[t], action[t+1], reward[t+1], obs[t+1], done[t+1])`.

`terminated` and `truncated` are separate signals, exactly as in Gymnasium.
D4RL-style "action-out" data (action paired with the observation it was
taken *at*) is converted once at the write boundary — see the README's
[Action-out data](../README.md#action-out-data) section — so everything
downstream only ever sees one convention.

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
    "terminated": True,
}]

dataset = Dataset.from_episodes(episodes)  # schema inferred automatically
```

## Reading data

```python
seg = dataset.episode(0).segment(0, 8)   # steps [0, 8) of episode 0

seg["front_camera"]      # flat field access
seg.image.front_camera   # space access — grouped by shared shape/dtype
seg.action, seg.reward   # bare action/reward arrays resolve directly
seg.observations         # role view: every observation-role field
seg.terminated           # [L] flags, True only on a terminal final step
```

Fields, spaces, and roles are just different ways of naming the same flat
storage — pick whichever reads best at each call site.

## Sampling for training

### Map-style: `segments()` + `DataLoader`

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
training:

```python
stream = dataset.segment_stream(
    fields=["front_camera", "state", "action"],
    context_length=4, target_length=32,   # or sequence_length=N for one block
    batch_size=64, seed=0,
)
batch = stream.sample()          # arrays [B, L, ...]
batch.context, batch.target      # time-sliced views

transitions = dataset.sample_transitions(batch_size=256)
transitions.observations, transitions.actions, transitions.next_observations
```

## Persisting and scaling up

```python
dataset = Dataset.from_episodes(episodes, path="my_dataset")  # npz_directory
dataset = Dataset.open("my_dataset")                          # reopen anywhere

# Outgrew one-file-per-episode? Stream across the storage boundary:
big = Dataset.open("my_dataset").copy_to("my_dataset_zarr", backend="zarr")
```

## Collecting data online, two ways

The write API mirrors a Gymnasium rollout one call per `env.step`. The usual
way is a writer, kept for the lifetime of the rollout:

```python
writer = dataset.new_episode()
writer.add_reset(obs, infos=info)

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
episode_id = dataset.new_episode().episode_id   # writer discarded; only the id is kept

dataset.add_reset(episode_id, obs)
dataset.add_steps(episode_id, {                 # a whole segment, one call
    "observations": obs_segment, "actions": action_segment,
    "rewards": reward_segment, "terminated": terminated_segment,
})
dataset.end_episode(episode_id, terminated=True)  # or let a True signal finalize it
```

## Next steps

- [`examples/getting_started.ipynb`](../examples/getting_started.ipynb) — a
  runnable tour of everything above.
- [README](../README.md) — full API reference: hierarchical fields,
  schema refinement, backend internals, action-out conversion.
