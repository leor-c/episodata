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

`segment_stream` is an infinite, shuffled stream. Need `DataLoader`-style
multi-worker reads instead? Use `dataset.segments(...)`, an indexable,
map-style dataset — see the README's
[Map-style access](../README.md#map-style-access-torchutilsdatadataloader).

## Persisting and scaling up

```python
dataset = Dataset.from_episodes(episodes, path="my_dataset")  # npz_directory
dataset = Dataset.open("my_dataset")                          # reopen anywhere

# Outgrew one-file-per-episode? Stream across the storage boundary:
big = Dataset.open("my_dataset").copy_to("my_dataset_zarr", backend="zarr")
```

## Collecting data online

The write API mirrors a Gymnasium rollout one call per `env.step`:

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

## Next steps

- [`examples/getting_started.ipynb`](../examples/getting_started.ipynb) — a
  runnable tour of everything above.
- [README](../README.md) — full API reference: hierarchical fields,
  schema refinement, backend internals, action-out conversion.
