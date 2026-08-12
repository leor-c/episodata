# Episodata

Episodic data library for world models, control, and reinforcement learning.

One data model and infrastructure that works for toy projects and scales to large
multimodal datasets. The underlying storage stays transparent: choose it up
front or migrate later without changing the API.

World-model repositories repeatedly rebuild the same episode storage,
alignment, sampling, and batching machinery. Episodata aims to make that
shared infrastructure: one well-tested implementation that reduces duplicated
engineering and the subtle bugs it creates.

Whether collecting live rollouts or importing existing data, model code reads
and samples through the same transition-aligned API, independent of storage.


```text
schema and episodes  ->  transition views and sampling  ->  storage backend
```

Public data is exposed as NumPy arrays. Persistent datasets use chunked Zarr v3
storage through TensorStore; PyTorch, TensorDict, and Gymnasium integrations
remain optional.

## Install

```bash
pip install episodata
pip install "episodata[gym]"       # optional Gymnasium schema helper
```

Python 3.11 or newer is required.

## Quick start

Start with the environment's schema, then collect with the same reset/step loop
used to run it:

```python
import gymnasium as gym
from episodata import Dataset
from episodata.utils import schema_from_gym_spaces

env = gym.make("CartPole-v1")
schema = schema_from_gym_spaces(env.observation_space, env.action_space)
dataset = Dataset.create(schema, path="cartpole")

obs, _ = env.reset()
writer = dataset.new_episode(obs)

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    writer.add_step({
        "observations": obs,
        "actions": action,
        "rewards": reward,
        "terminated": terminated,
        "truncated": truncated,
    })
    if terminated or truncated:       # the writer is already finalized
        break
```

Every read returns transitions, with the pairing made explicit:

```python
segment = dataset.episode(0).read()

segment.obs              # observation where each action was taken: [T, 4]
segment.action           # action taken: [T]
segment.reward           # reward received: [T]
segment.next_obs         # observation produced: [T, 4]
segment.terminated       # terminal flag per transition: [T]
```

Sample fixed-length batches for training:

```python
stream = dataset.segment_stream(sequence_length=32, batch_size=64, seed=0)
batch = stream.sample()

batch.obs                # [64, 32, 4]
batch.next_obs           # [64, 32, 4]
batch.mask               # [64, 32], False where a short episode was padded
```

## The data model

An episode with `T` environment steps has length `T` and contains exactly `T`
transitions. A segment of `L` transitions exposes:

| Value | Meaning | Shape |
|---|---|---|
| `observation` / `obs` | observation where the action was taken | `[L, ...]` |
| `action`, `reward` | action and resulting reward | `[L, ...]` |
| `next_observation` / `next_obs` | observation produced by the action | `[L, ...]` |
| `terminated`, `truncated`, `mask` | per-transition flags | `[L]` |
| `all_observations` / `all_obs` | complete observation sequence | `[L + 1, ...]` |

`observation` and `next_observation` are overlapping views of the same
`L + 1` row buffer. Consecutive observations are not duplicated.

Fields are accessed role first. A role created from one array unwraps to that
array; a dict-shaped role remains a mapping:

```python
episode = {
    "initial_observation": {"camera": frame0, "state": state0},
    "observations": {"camera": frames, "state": states},
    "actions": {"move": moves, "camera": camera_actions},
    "rewards": rewards,
}
dataset = Dataset.from_episodes([episode])
segment = dataset.episode(0).read()

segment.obs.camera
segment.action.move
segment.obs["camera"]
```

Nested dicts use stable `/`-separated keys in the schema and storage layer,
while attribute access reconstructs the hierarchy:

```python
segment.action["keyboard/jump"]
segment.action.keyboard.jump
```

## Other creation paths

### Declare a schema directly

When there is no environment to inspect, declare the schema before collecting
data:

```python
from episodata import Dataset, DatasetSchema, FieldSpec

schema = DatasetSchema([
    FieldSpec("camera", shape=(3, 64, 64), dtype="uint8", layout="CHW"),
    FieldSpec("state", shape=(7,), dtype="float32"),
    FieldSpec("action", shape=(4,), dtype="float32", role="action"),
    FieldSpec("reward", shape=(), dtype="float32", role="reward"),
])
dataset = Dataset.create(schema)
```

The schema is authoritative after creation: it defines stable field keys,
per-step shapes, dtypes, roles, and optional metadata independently of physical
storage.

### Import existing episodes

Use `initial_observation` for the reset observation and one entry per
environment step for every temporal field:

```python
import numpy as np

episode = {
    "initial_observation": np.zeros(4, dtype=np.float32),
    "observations": np.zeros((100, 4), dtype=np.float32),
    "actions": np.zeros(100, dtype=np.int64),
    "rewards": np.zeros(100, dtype=np.float32),
    "terminated": True,
}

dataset = Dataset.from_episodes([episode])                       # memory
dataset = Dataset.from_episodes([episode], path="rollouts")      # zarr
dataset = Dataset.open("rollouts")
```

The schema is inferred from the first episode unless supplied explicitly.

## Sampling

### PyTorch `DataLoader`

`segments()` is an indexable map-style dataset. Its batched-fetch hook turns a
whole loader batch into one backend read, and workers can read independently.
Episodata itself does not import PyTorch.

```python
import torch
from torch.utils.data import DataLoader

segments = dataset.segments(sequence_length=32)
loader = DataLoader(
    segments,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    collate_fn=segments.collate,
)

for batch in loader:
    batch = batch.map(lambda x: torch.as_tensor(x).to("cuda"))
```

`map()` converts each underlying field buffer once, preserving the shared
storage between `obs` and `next_obs` after a device transfer.

### Streaming and context/target batches

`segment_stream()` is an infinite uniform-with-replacement stream by default.
It reads large chunks internally and buffers ready batches to amortize backend
overhead.

```python
stream = dataset.segment_stream(
    context_length=8,
    target_length=24,
    batch_size=64,
    seed=0,
)

batch = stream.sample()
batch.context             # [B, 8, ...]
batch.target              # [B, 24, ...]
```

Other common queries:

```python
# One-shot transitions, with the time axis removed
transitions = dataset.sample_transitions(batch_size=256)

# One finite pass over all valid segments
for batch in dataset.segment_stream(sequence_length=32, batch_size=64, shuffle=False):
    ...

# Only episodes accepted by the predicate
stream = dataset.segment_stream(
    sequence_length=32,
    filter=lambda episode: episode.terminated,
)
```

Short episodes produce one zero-padded segment by default. Set `pad="prefix"`
to pad at the beginning or `pad=None` to skip them. `batch.mask` always marks
real transitions.

Custom sampling policies implement one method and can be passed directly to
`segment_stream()`:

```python
import numpy as np

class MySampler:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def sample(self, index, batch_size):
        return self.rng.integers(len(index), size=batch_size)

stream = dataset.segment_stream(
    sequence_length=32,
    batch_size=64,
    sampler=MySampler(seed=0),
)
```

## TensorDict interop

```python
from episodata.utils import batch_to_tensordict

td = batch_to_tensordict(batch, device="cuda")
```

Install this optional integration with `pip install torch tensordict`. Passing
the device to `batch_to_tensordict` preserves the shared observation storage;
moving the returned TensorDict afterward may copy its role views separately.

## Storage

| Backend | Use case |
|---|---|
| `memory` | tests, small datasets, replay buffers; default without `path` |
| `zarr` | chunked persistent datasets; default with `path` |
| `npz_directory` | simple inspectable files, one compressed archive per episode |

The Zarr backend stores standard Zarr v3 arrays and uses TensorStore as its I/O
engine. Sampling paths issue batched reads across selections. The NPZ backend is
simple, but reading a small segment still decompresses its episode archive.

Opening reads the backend from the manifest. Migration does not change the
query API:

```python
source = Dataset.open("old_npz_dataset")
dataset = source.copy_to("chunked_dataset", backend="zarr")
```


Internally, episodes are stored in action-in form with a reset row. Action-out
sources such as D4RL are converted once when written; every read uses the same
transition API.

## Documentation

- [Getting started](docs/getting_started.md) — practical workflows and edge cases
- [API reference](docs/api.md) — public types, signatures, and contracts
- [Runnable notebook](examples/getting_started.ipynb) — an interactive tour
