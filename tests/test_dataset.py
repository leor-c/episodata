import numpy as np
import pytest

from episodata import Dataset, DatasetSchema
from tests.conftest import make_episode, make_steps


def test_basic_properties(dataset):
    assert dataset.num_episodes == 2
    episode = dataset.episode(0)
    assert len(episode) == 10
    assert episode.terminated and not episode.truncated and not episode.ongoing
    assert not dataset.episode(1).terminated


def test_segment_matches_source(dataset):
    source = make_episode(10, seed=0)
    segment = dataset.episode(0).segment(2, 6, fields=["front_camera", "reward"])
    # transition t pairs the obs the action was taken at (source obs t - 1)
    # with the action/reward of step t and the obs it produced (source obs t)
    assert np.array_equal(segment.obs["front_camera"], source["observations"]["front_camera"][1:5])
    assert np.array_equal(
        segment.next_obs["front_camera"], source["observations"]["front_camera"][2:6]
    )
    assert np.array_equal(segment.reward, source["rewards"][2:6])
    step = dataset.episode(0).step(-1)
    assert np.array_equal(step.obs["state"], source["observations"]["state"][-2])
    assert np.array_equal(step.next_obs["state"], source["observations"]["state"][-1])
    assert np.array_equal(step.action, source["actions"]["action"][-1])


def test_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "ds")
    original = Dataset.from_episodes([make_episode(6)], path=path)
    reopened = Dataset.open(path)
    assert reopened.schema.to_dict() == original.schema.to_dict()
    assert reopened.num_episodes == 1
    assert np.array_equal(
        reopened.episode(0).read().obs["state"], original.episode(0).read().obs["state"]
    )


def test_persistence_roundtrip_accepts_path_object(tmp_path):
    # `path` must also accept a pathlib.Path, not just str.
    path = tmp_path / "ds"
    original = Dataset.from_episodes([make_episode(6)], path=path)
    reopened = Dataset.open(path)
    assert reopened.schema.to_dict() == original.schema.to_dict()
    assert reopened.num_episodes == 1
    assert np.array_equal(
        reopened.episode(0).read().obs["state"], original.episode(0).read().obs["state"]
    )


def test_schema_not_reinferred_on_open(tmp_path):
    # Declared bounds inference would never produce: seeing them after
    # reopen proves the persisted schema is read, not re-inferred.
    path = str(tmp_path / "ds")
    schema = DatasetSchema.infer(make_episode(6))
    schema.field("state").low = -1.0
    schema.field("state").high = 1.0
    Dataset.from_episodes([make_episode(6)], schema=schema, path=path)
    reopened = Dataset.open(path)
    assert reopened.schema.field("state").low == -1.0
    assert reopened.schema.field("state").high == 1.0
    seg = reopened.episode(0).segment(0, 2)
    assert seg.obs.state.shape == (2, 5)


def test_online_append(backend_name, dataset_path):
    schema = DatasetSchema.infer(make_episode(3))
    dataset = Dataset.create(schema, path=dataset_path, backend=backend_name)
    assert dataset.num_episodes == 0

    writer = dataset.new_episode(
        {
            "front_camera": np.zeros((3, 8, 8), dtype=np.uint8),
            "wrist_camera": np.zeros((3, 8, 8), dtype=np.uint8),
            "state": np.zeros(5, dtype=np.float32),
        }
    )
    episode = dataset.episode(writer.episode_id)
    for t in range(5):
        writer.add_step(
            {
                "observations": {
                    "front_camera": np.full((3, 8, 8), t, dtype=np.uint8),
                    "wrist_camera": np.zeros((3, 8, 8), dtype=np.uint8),
                    "state": np.zeros(5, dtype=np.float32),
                },
                "actions": {"action": np.zeros(2, dtype=np.float32)},
                "rewards": float(t),
            }
        )
    assert episode.ongoing and len(episode) == 5
    # ongoing episodes are readable
    assert np.array_equal(episode.segment(1, 3).reward, [1.0, 2.0])

    writer.end(terminated=True)
    assert not episode.ongoing and episode.terminated
    with pytest.raises(ValueError):
        writer.add_step({"rewards": 0.0})


def _step(reward: float = 0.0) -> dict:
    return {
        "observations": {
            "front_camera": np.zeros((3, 8, 8), dtype=np.uint8),
            "wrist_camera": np.zeros((3, 8, 8), dtype=np.uint8),
            "state": np.zeros(5, dtype=np.float32),
        },
        "actions": np.zeros(2, dtype=np.float32),
        "rewards": reward,
    }


def _reset_obs() -> dict:
    return {
        "front_camera": np.full((3, 8, 8), 5, dtype=np.uint8),
        "wrist_camera": np.zeros((3, 8, 8), dtype=np.uint8),
        "state": np.ones(5, dtype=np.float32),
    }


def test_new_episode_writes_reset_row(dataset):
    writer = dataset.new_episode(_reset_obs())
    # the reset observation alone is no transition yet
    assert len(dataset.episode(writer.episode_id)) == 0
    writer.add_step(_step(1.0))
    row = dataset.episode(writer.episode_id).step(0)
    assert np.array_equal(row.obs["state"], np.ones(5, dtype=np.float32))  # reset obs
    assert np.array_equal(row.next_obs["state"], np.zeros(5, dtype=np.float32))
    assert np.array_equal(row.action, np.zeros(2, dtype=np.float32))
    assert row.reward == 1.0
    # infos accompany the reset observation, never stand alone
    with pytest.raises(ValueError, match="observations"):
        dataset.new_episode(infos={"success": False})


def test_gymnasium_style_step_signals(dataset):
    writer = dataset.new_episode(_reset_obs())
    writer.add_step({**_step(1.0), "terminated": False, "truncated": False})
    writer.add_step({**_step(2.0), "terminated": True, "truncated": False})

    episode = dataset.episode(writer.episode_id)
    assert not episode.ongoing and episode.terminated and not episode.truncated
    assert len(episode) == 2
    # the terminal signal closed the writer, Gymnasium-style
    with pytest.raises(ValueError, match="closed"):
        writer.add_step(_step())


def test_truncated_step_signal(dataset):
    writer = dataset.new_episode(_reset_obs())
    writer.add_step({**_step(), "truncated": True})
    episode = dataset.episode(writer.episode_id)
    assert episode.truncated and not episode.terminated and not episode.ongoing


def test_segment_with_terminal_flag_finalizes(dataset):
    writer = dataset.new_episode(_reset_obs())
    writer.add_steps(make_steps(3, seed=2, terminated=True))
    assert dataset.episode(writer.episode_id).terminated


def test_flags_validated_against_position_and_length():
    with pytest.raises(ValueError, match="final step"):
        Dataset.from_episodes(
            [{**make_episode(5), "terminated": [False, True, False, False, False]}]
        )
    with pytest.raises(ValueError, match="length"):
        Dataset.from_episodes([{**make_episode(5), "terminated": [False, True]}])


def test_resume_episode(dataset):
    episode_id = dataset.new_episode(_reset_obs()).episode_id
    # later, without the original writer: reattach by id ...
    writer = dataset.resume_episode(episode_id)
    writer.add_step(_step(1.0))
    # ... or via the episode view
    dataset.episode(episode_id).writer().add_step(_step(2.0))
    assert np.array_equal(dataset.episode(episode_id).read().reward, [1.0, 2.0])

    writer.end(terminated=True)
    with pytest.raises(ValueError, match="finalized"):
        dataset.resume_episode(episode_id)


def test_direct_id_based_append(dataset):
    episode_id = dataset.new_episode(_reset_obs()).episode_id
    dataset.add_step(episode_id, _step(1.0))
    dataset.add_steps(
        episode_id,
        {
            "observations": {
                "front_camera": np.zeros((2, 3, 8, 8), dtype=np.uint8),
                "wrist_camera": np.zeros((2, 3, 8, 8), dtype=np.uint8),
                "state": np.zeros((2, 5), dtype=np.float32),
            },
            "actions": np.zeros((2, 2), dtype=np.float32),
            "rewards": [2.0, 3.0],
        },
    )
    episode = dataset.end_episode(episode_id, terminated=True)
    assert episode.terminated and len(episode) == 3
    assert np.array_equal(episode.read().reward, [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="finalized"):
        dataset.add_step(episode_id, _step())
    with pytest.raises(ValueError, match="finalized"):
        dataset.end_episode(episode_id)


def test_resume_after_reopen(tmp_path):
    path = str(tmp_path / "ds")
    dataset = Dataset.from_episodes([make_episode(4)], path=path)
    writer = dataset.new_episode(make_episode(1, seed=7)["initial_observation"])
    writer.add_steps(make_steps(3, seed=7, terminated=False))
    episode_id = writer.episode_id
    dataset.flush()

    reopened = Dataset.open(path)
    writer = reopened.resume_episode(episode_id)
    writer.add_step(_step(9.0))
    episode = writer.end(terminated=True)
    assert len(episode) == 4 and episode.terminated
    assert episode.read().reward[-1] == 9.0


def test_append_validation(dataset):
    writer = dataset.new_episode()
    with pytest.raises(ValueError, match="shape"):
        writer.add_step(
            {
                "observations": {
                    "front_camera": np.zeros((3, 4, 4), dtype=np.uint8),
                    "wrist_camera": np.zeros((3, 8, 8), dtype=np.uint8),
                    "state": np.zeros(5),
                },
                "actions": np.zeros(2),
                "rewards": 0.0,
            }
        )
    with pytest.raises(ValueError, match="missing required"):
        writer.add_step({"rewards": 0.0})


def test_ongoing_episode_survives_reopen(tmp_path):
    path = str(tmp_path / "ds")
    dataset = Dataset.from_episodes([make_episode(4)], path=path)
    writer = dataset.new_episode(make_episode(1, seed=7)["initial_observation"])
    writer.add_steps(make_steps(3, seed=7, terminated=False))
    dataset.flush()

    reopened = Dataset.open(path)
    episode = reopened.episode(writer.episode_id)
    assert episode.ongoing and len(episode) == 3
    source = make_steps(3, seed=7)
    assert np.array_equal(
        episode.read().next_obs["state"], source["observations"]["state"]
    )


def test_declared_schema_mode(backend_name, dataset_path):
    schema = DatasetSchema.infer(make_episode(3))
    schema.field("state").low = -1.0
    dataset = Dataset.from_episodes(
        [make_episode(5)], schema=schema, path=dataset_path, backend=backend_name
    )
    assert dataset.schema.field("state").low == -1.0
    assert dataset.episode(0).segment(0, 2).obs.state.shape == (2, 5)


def test_append_to_finalized_episode_says_so(dataset):
    # the finalized check runs before step normalization, so even a
    # malformed step gets the real reason
    episode_id = dataset.episode(0).id  # episode 0 is finalized
    with pytest.raises(ValueError, match="finalized"):
        dataset.add_step(episode_id, {})
    with pytest.raises(ValueError, match="finalized"):
        dataset.add_steps(episode_id, {})
