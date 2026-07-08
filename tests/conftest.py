import importlib.util

import numpy as np
import pytest

from episodata import Dataset

requires_zarr = pytest.mark.skipif(
    importlib.util.find_spec("zarr") is None, reason="zarr not installed"
)


def make_episode(length: int, seed: int = 0, terminated: bool = True):
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


@pytest.fixture(params=["memory", "npz_directory", pytest.param("zarr", marks=requires_zarr)])
def backend_name(request):
    return request.param


@pytest.fixture
def dataset_path(backend_name, tmp_path):
    return None if backend_name == "memory" else str(tmp_path / "ds")


@pytest.fixture
def dataset(backend_name, dataset_path):
    episodes = [make_episode(10, seed=0), make_episode(7, seed=1, terminated=False)]
    return Dataset.from_episodes(episodes, path=dataset_path, backend=backend_name)