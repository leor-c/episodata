# Getting started

This guide covers the few concepts needed to use Episodata confidently. See
the [API reference](api.md) for complete signatures and the
[notebook](../examples/getting_started.ipynb) for a runnable tour.

## Install

```bash
pip install episodata
pip install "episodata[gym]"       # optional: Gymnasium schema conversion
```

The core package requires Python 3.11 or newer and exposes NumPy arrays.

## Mental model

Episodata separates three concerns:

```text
DatasetSchema  ->  Episode / Segment / Batch  ->  StorageBackend
 meaning             queries and sampling           persistence
```

A schema describes each field's role, per-step shape, and dtype. An episode
contains `T` environment steps. Reads return `T` transitions, regardless of
the backend.

For a segment of `L` transitions:

```text
observations:      o0  o1  ...  oL-1
actions/rewards:   a0  a1  ...  aL-1
next_observations: o1  o2  ...  oL
all_observations:  o0  o1  ...  oL-1  oL
```

Thus `segment.obs[i]`, `segment.action[i]`, `segment.reward[i]`, and
`segment.next_obs[i]` always describe one transition. The two observation
views overlap in one `L + 1` row buffer.

## 1. Create a dataset

### Before data exists

Create a schema from Gymnasium spaces:

```python
import gymnasium as gym
from episodata import Dataset
from episodata.utils import schema_from_gym_spaces

env = gym.make("CartPole-v1")
schema = schema_from_gym_spaces(env.observation_space, env.action_space)
dataset = Dataset.create(schema, path="cartpole")
```

Or declare it directly:

```python
from episodata import Dataset, DatasetSchema, FieldSpec

schema = DatasetSchema([
    FieldSpec("image", shape=(64, 64, 3), dtype="uint8", layout="HWC"),
    FieldSpec("state", shape=(12,), dtype="float32"),
    FieldSpec("control", shape=(4,), dtype="float32", role="action"),
    FieldSpec("reward", shape=(), dtype="float32", role="reward"),
])
dataset = Dataset.create(schema)
```

`shape` never includes a time or batch dimension. The supported roles are
`observation`, `action`, `reward`, and `info`; the default is `observation`.

### From existing episodes

Use `initial_observation` for the reset observation and one entry per
environment step for every temporal field:

```python
import numpy as np
from episodata import Dataset

episodes = [{
    "initial_observation": np.zeros(6, dtype=np.float32),
    "observations": np.zeros((20, 6), dtype=np.float32),
    "actions": np.zeros((20, 2), dtype=np.float32),
    "rewards": np.zeros(20, dtype=np.float32),
    "terminated": True,
    "truncated": False,
}]

dataset = Dataset.from_episodes(episodes)                       # memory
dataset = Dataset.from_episodes(episodes, path="training_data") # zarr
```

The schema is inferred from the first episode unless supplied explicitly.
After creation, the schema is authoritative and persisted with the dataset.

## 2. Write episodes

The online API mirrors a Gymnasium rollout. `new_episode()` records the reset
observation, and each `add_step()` records one call to `env.step()`:

```python
obs, _ = env.reset()
writer = dataset.new_episode(obs)

while True:
    action = policy(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    writer.add_step({
        "observations": obs,
        "actions": action,
        "rewards": reward,
        "terminated": terminated,
        "truncated": truncated,
    })
    if terminated or truncated:
        break
```

A true termination signal finalizes the episode. Otherwise close it explicitly:

```python
episode = writer.end(truncated=True)
```

Writers are stateless handles. An ongoing episode can be resumed by ID, even
after reopening persistent storage:

```python
episode_id = writer.episode_id
dataset.flush()

dataset = Dataset.open("training_data")
writer = dataset.resume_episode(episode_id)
```

Use `writer.add_steps(segment)` to append several steps with a leading time
dimension. Equivalent ID-based methods are available on `Dataset`.

### Vectorized environments

`VectorWriter` accepts arrays with a leading environment dimension and tracks
one episode per environment. It implements next-step autoreset semantics: after
an environment finishes, its next observation begins the next episode.

```python
writer = dataset.vector_writer()
obs, _ = envs.reset()
writer.reset(obs)

for _ in range(num_steps):
    actions = policy(obs)
    obs, rewards, terminated, truncated, _ = envs.step(actions)
    writer.step(
        obs,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
    )

writer.close()             # remaining episodes become truncated
```

For same-step autoreset environments, keep one ordinary `EpisodeWriter` per
environment and write the true final observation from the environment's info.

## 3. Read episodes

`Episode` is lazy: data is read only when `read`, `segment`, or `step` is
called.

```python
episode = dataset.episode(0)

len(episode)                       # number of transitions
episode.read()                     # full episode
episode.segment(4, 12)             # transitions [4, 12)
transition = episode.step(4)       # time dimension removed
```

Role-first access works uniformly for segments and batches:

```python
segment = episode.segment(4, 12)

segment.obs.state
segment.next_obs.state
segment.action
segment.reward
segment.terminated
```

A role backed by a single array unwraps to that array. Dict observations or
actions return a mapping with item and attribute access. Nested fields are
flattened to `/`-separated schema keys:

```python
segment.action["keyboard/jump"]
segment.action.keyboard.jump
```

Field selection accepts exact keys or group prefixes:

```python
segment = episode.read(fields=["image", "keyboard"])
```

## 4. Sample training data

### Map-style dataset

Use `segments()` with PyTorch's `DataLoader` when you want its shuffling,
samplers, and worker processes:

```python
from torch.utils.data import DataLoader

segments = dataset.segments(sequence_length=16)
loader = DataLoader(
    segments,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    collate_fn=segments.collate,
)

for batch in loader:
    ...
```

`segments[i]` returns one `Segment`. DataLoader automatically uses Episodata's
batched-fetch hook, so a loader batch requires one backend read rather than one
read per segment. On a growing dataset, call `segments.refresh()` between
epochs; do not change its index during an epoch.

### Streaming sampler

Use `segment_stream()` for an infinite, single-process stream:

```python
stream = dataset.segment_stream(
    context_length=8,
    target_length=24,
    batch_size=64,
    seed=0,
)

batch = stream.sample()
batch.context
batch.target
```

Sampling is uniform over all valid `(episode, start)` windows, with replacement.
The stream fetches `max(batch_size, 2048)` segments per backend call by default
and buffers the resulting batches. Set `read_chunk_size` to tune throughput and
memory use. On a growing dataset, new episodes become visible at the next
buffer refill.

A custom stream sampler only needs to return flat segment indices:

```python
import numpy as np

class WeightedSampler:
    def __init__(self, weights, seed=None):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def sample(self, index, batch_size):
        probabilities = self.weights / self.weights.sum()
        return self.rng.choice(len(index), size=batch_size, p=probabilities)

stream = dataset.segment_stream(
    sequence_length=16,
    batch_size=64,
    sampler=WeightedSampler(priorities, seed=0),
)
```

For a finite sequential pass, iterate a stream with `shuffle=False`. For
individual random transitions, use `dataset.sample_transitions(batch_size)`.

### Padding

An episode shorter than the requested sequence contributes one segment:

- `pad="suffix"` pads its end; this is the default.
- `pad="prefix"` pads its beginning.
- `pad=None` excludes it.

`batch.mask` is true only for real transitions. Terminal and truncated flags
remain aligned with the episode's real final transition.

## 5. Convert batches

Use `map()` for any conversion that may copy data. It converts each underlying
field buffer once and then rebuilds the overlapping role views:

```python
batch = batch.map(lambda array: torch.as_tensor(array).to("cuda"))
```

For TensorDict:

```python
from episodata.utils import batch_to_tensordict

td = batch_to_tensordict(batch, device="cuda")
```

Install the optional packages with `pip install torch tensordict`. Use
`include_all_observations=True` to include the `L + 1` sequence. Use
`alignment="action_in"` or `alignment="action_out"` only when a downstream
TensorDict consumer requires every entry to have length `L + 1`; the missing
action/reward/flag row is zero-filled.

## Action-out sources

Sources such as D4RL often store the action taken *at* each observation. Mark a
bulk episode with `final_observation`; Episodata converts it at the write
boundary:

```python
episode = {
    "observations": observations,
    "actions": actions,
    "rewards": rewards,
    "final_observation": final_observation,
    "terminated": True,
}
dataset = Dataset.from_episodes([episode])
```

For streaming action-out data, wrap an empty writer:

```python
from episodata import ActionOutWriter

writer = ActionOutWriter(dataset.new_episode())
writer.add_step({"observations": obs, "actions": action, "rewards": reward})
episode = writer.end(terminated=True, final_observation=final_obs)
```

Every subsequent read uses the ordinary transition API; alignment is never a
read-time concern.

## Storage choices

```python
Dataset.create(schema)                                      # memory
Dataset.create(schema, path="data")                         # zarr
Dataset.create(schema, path="data", backend="npz_directory")
Dataset.open("data")                                        # backend from manifest
```

Choose `zarr` for long episodes, large observations, or random segment reads.
It stores standard Zarr v3 arrays and uses TensorStore for batched I/O. Choose
`npz_directory` when simple per-episode archives matter more than partial-read
performance.

Migrate through the logical API:

```python
Dataset.open("npz_data").copy_to("zarr_data", backend="zarr")
```
