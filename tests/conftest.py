import importlib.util

import numpy as np
import pytest

from episodata import Dataset

requires_gym = pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None, reason="gymnasium not installed"
)
requires_tensordict = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("tensordict") is None,
    reason="torch/tensordict not installed",
)


def make_episode(length: int, seed: int = 0, terminated: bool = True):
    """Build a bulk-import episode dict with ``length`` env steps:
    ``initial_observation`` (the reset observation, no time dim) plus
    equal-length observations/actions/rewards — one entry per step."""
    rng = np.random.default_rng(seed)
    return {
        "initial_observation": {
            "front_camera": rng.integers(0, 256, size=(3, 8, 8), dtype=np.uint8),
            "wrist_camera": rng.integers(0, 256, size=(3, 8, 8), dtype=np.uint8),
            "state": rng.standard_normal(5).astype(np.float32),
        },
        "observations": {
            "front_camera": rng.integers(0, 256, size=(length, 3, 8, 8), dtype=np.uint8),
            "wrist_camera": rng.integers(0, 256, size=(length, 3, 8, 8), dtype=np.uint8),
            "state": rng.standard_normal((length, 5)).astype(np.float32),
        },
        "actions": {"action": rng.standard_normal((length, 2)).astype(np.float32)},
        "rewards": rng.standard_normal(length).astype(np.float32),
        "terminated": terminated,
    }


def make_steps(length: int, seed: int = 0, terminated: bool = True):
    """Build a plain step-batch dict (no reset row) for continuing an
    already-open episode via ``writer.add_steps``/``Dataset.add_steps``:
    observations, actions, rewards, terminated all share ``length``."""
    rng = np.random.default_rng(seed)
    return {
        "observations": {
            "front_camera": rng.integers(0, 256, size=(length, 3, 8, 8), dtype=np.uint8),
            "wrist_camera": rng.integers(0, 256, size=(length, 3, 8, 8), dtype=np.uint8),
            "state": rng.standard_normal((length, 5)).astype(np.float32),
        },
        "actions": {"action": rng.standard_normal((length, 2)).astype(np.float32)},
        "rewards": rng.standard_normal(length).astype(np.float32),
        "terminated": terminated,
    }


@pytest.fixture(params=["memory", "npz_directory", "zarr"])
def backend_name(request):
    return request.param


@pytest.fixture
def dataset_path(backend_name, tmp_path):
    return None if backend_name == "memory" else str(tmp_path / "ds")


@pytest.fixture
def dataset(backend_name, dataset_path):
    episodes = [make_episode(10, seed=0), make_episode(7, seed=1, terminated=False)]
    return Dataset.from_episodes(episodes, path=dataset_path, backend=backend_name)