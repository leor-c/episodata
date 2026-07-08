"""Hierarchical (nested) fields: Minecraft-style complex actions and
structured observations, flattened to path keys with hierarchy
reconstructed at the access layer."""

import numpy as np
import pytest

from episodata import Dataset, DatasetSchema


def minecraft_episode(length: int = 8) -> dict:
    t = np.arange(length)
    return {
        "observations": {
            "pov": np.zeros((length, 3, 16, 16), dtype=np.uint8),
            "inventory": {
                "stone": t.astype(np.int64),
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
    obs = dataset.episode(0).read()
    assert np.array_equal(obs.keyboard.w, np.ones(8, dtype=np.uint8))
    assert np.array_equal(obs["keyboard/w"], obs["keyboard"]["w"])
    assert set(obs.keyboard) == {"w", "jump"}
    assert set(dict(obs.inventory.items())) == {"stone", "wood"}
    assert np.array_equal(obs.inventory.stone, np.arange(8))
    with pytest.raises(AttributeError, match="keyboard"):
        _ = obs.keyboard.missing


def test_prefix_field_selection(backend_name, dataset_path):
    dataset = Dataset.from_episodes(
        [minecraft_episode()], path=dataset_path, backend=backend_name
    )
    stream = dataset.segment_stream(
        fields=["pov", "keyboard"], sequence_length=3, batch_size=2, seed=0
    )
    batch = stream.sample()
    assert set(batch.keys()) == {"pov", "keyboard/w", "keyboard/jump"}
    assert batch.keyboard.w.shape == (2, 3)


def test_transitions_group_access():
    dataset = Dataset.from_episodes([minecraft_episode()])
    transitions = dataset.sample_transitions(batch_size=4, seed=0)
    assert transitions.actions.keyboard.w.shape == (4,)
    assert transitions.observations.inventory.stone.shape == (4,)


def test_nested_online_append(backend_name, dataset_path):
    schema = DatasetSchema.infer(minecraft_episode())
    dataset = Dataset.create(schema, path=dataset_path, backend=backend_name)
    writer = dataset.new_episode()
    writer.add_reset(
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
    assert len(episode) == 2 and episode.terminated
    data = episode.read()
    assert np.array_equal(data["keyboard/w"], [0, 1])
    assert np.array_equal(data["inventory/stone"], [0, 1])
