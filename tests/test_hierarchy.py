"""Hierarchical (nested) fields: Minecraft-style complex actions and
structured observations, flattened to path keys with hierarchy
reconstructed at the access layer."""

import numpy as np
import pytest

from episodata import Dataset, DatasetSchema


def minecraft_episode(length: int = 8) -> dict:
    """``length`` env steps: ``initial_observation`` plus equal-length
    observations/actions/rewards. Stone count equals the step number."""
    return {
        "initial_observation": {
            "pov": np.zeros((3, 16, 16), dtype=np.uint8),
            "inventory": {"stone": 0, "wood": 0},
        },
        "observations": {
            "pov": np.zeros((length, 3, 16, 16), dtype=np.uint8),
            "inventory": {
                "stone": np.arange(1, length + 1, dtype=np.int64),
                "wood": np.zeros(length, dtype=np.int64),
            },
        },
        "actions": {
            "camera": np.zeros((length, 2), dtype=np.float32),
            "keyboard": {
                "w": np.ones(length, dtype=np.uint8),
                "jump": np.zeros(length, dtype=np.uint8),
            },
        },
        "rewards": np.zeros(length, dtype=np.float32),
        "terminated": True,
    }


def test_nested_ingest_flattens_to_path_keys(backend_name, dataset_path):
    dataset = Dataset.from_episodes(
        [minecraft_episode()], path=dataset_path, backend=backend_name
    )
    assert "keyboard/w" in dataset.schema.fields
    assert "inventory/stone" in dataset.schema.fields
    assert dataset.schema.field("keyboard/w").role == "action"
    assert dataset.schema.field("inventory/stone").role == "observation"
    # structurally identical siblings still share a space
    assert dataset.schema.field("keyboard/w").space == dataset.schema.field("keyboard/jump").space


def test_group_access(backend_name, dataset_path):
    dataset = Dataset.from_episodes(
        [minecraft_episode()], path=dataset_path, backend=backend_name
    )
    seg = dataset.episode(0).read()
    assert np.array_equal(seg.action.keyboard.w, np.ones(8))
    assert np.array_equal(seg.action["keyboard/w"], seg.action["keyboard"]["w"])
    assert set(seg.action.keyboard) == {"w", "jump"}
    assert set(dict(seg.obs.inventory.items())) == {"stone", "wood"}
    # observations pair the obs each action was taken at: reset obs first
    assert np.array_equal(seg.obs.inventory.stone, np.arange(8))
    assert np.array_equal(seg.next_obs["inventory/stone"], np.arange(1, 9))
    with pytest.raises(AttributeError, match="keyboard"):
        _ = seg.action.keyboard.missing


def test_prefix_field_selection(backend_name, dataset_path):
    dataset = Dataset.from_episodes(
        [minecraft_episode()], path=dataset_path, backend=backend_name
    )
    stream = dataset.segment_stream(
        fields=["pov", "keyboard"], sequence_length=3, batch_size=2, seed=0
    )
    batch = stream.sample()
    assert set(batch.obs) == {"pov"}
    assert set(batch.action) == {"keyboard/w", "keyboard/jump"}
    assert batch.action.keyboard.w.shape == (2, 3)


def test_transitions_group_access():
    dataset = Dataset.from_episodes([minecraft_episode()])
    transitions = dataset.sample_transitions(batch_size=4, seed=0)
    assert transitions.actions.keyboard.w.shape == (4,)
    assert transitions.observations.inventory.stone.shape == (4,)


def test_nested_online_append(backend_name, dataset_path):
    schema = DatasetSchema.infer(minecraft_episode())
    dataset = Dataset.create(schema, path=dataset_path, backend=backend_name)
    writer = dataset.new_episode(
        {
            "pov": np.zeros((3, 16, 16), dtype=np.uint8),
            "inventory": {"stone": 0, "wood": 0},
        }
    )
    writer.add_step(
        {
            "observations": {
                "pov": np.zeros((3, 16, 16), dtype=np.uint8),
                "inventory": {"stone": 1, "wood": 0},
            },
            "actions": {
                "camera": np.zeros(2, dtype=np.float32),
                "keyboard": {"w": 1, "jump": 0},
            },
            "rewards": 0.5,
            "terminated": True,
        }
    )
    episode = dataset.episode(writer.episode_id)
    assert len(episode) == 1 and episode.terminated
    data = episode.read()
    assert np.array_equal(data.action["keyboard/w"], [1])
    assert np.array_equal(data.obs["inventory/stone"], [0])
    assert np.array_equal(data.next_obs["inventory/stone"], [1])
