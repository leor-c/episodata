"""VectorWriter: N parallel environments with staggered episode boundaries.

Next-step autoreset semantics throughout: when env ``i`` reports done at
step ``t``, its episode finalizes, and its observation at step ``t+1`` is
the reset row of a fresh episode (action/reward ignored).
"""

import numpy as np
import pytest

from episodata import Dataset, DatasetSchema


def _example():
    return {
        "initial_observation": {"state": np.zeros(3, dtype=np.float32)},
        "observations": {"state": np.zeros((2, 3), dtype=np.float32)},
        "actions": np.zeros((2, 2), dtype=np.float32),
        "rewards": np.zeros(2, dtype=np.float32),
    }


def make_dataset(backend_name, tmp_path, name="vec"):
    path = None if backend_name == "memory" else str(tmp_path / name)
    return Dataset.create(DatasetSchema.infer(_example()), path=path, backend=backend_name)


def obs(*vals):
    return {"state": np.array([[v] * 3 for v in vals], dtype=np.float32)}


def act(*vals):
    return np.array([[v] * 2 for v in vals], dtype=np.float32)


def rew(*vals):
    return np.array(vals, dtype=np.float32)


def write_reference_episode(
    dataset, reset_val, step_vals, terminated=False, truncated=False, end_truncated=False
):
    """The same trajectory written through the single-env API."""
    writer = dataset.new_episode({"state": np.full(3, reset_val, dtype=np.float32)})
    for j, v in enumerate(step_vals):
        last = j == len(step_vals) - 1
        writer.add_step(
            {
                "observations": {"state": np.full(3, v, dtype=np.float32)},
                "actions": np.full(2, v, dtype=np.float32),
                "rewards": np.float32(v),
                "terminated": terminated and last,
                "truncated": truncated and last,
            }
        )
    if dataset.episode(writer.episode_id).ongoing:
        writer.end(truncated=end_truncated)
    return writer.episode_id


def assert_episodes_equal(actual, expected):
    assert len(actual) == len(expected)
    assert actual.terminated == expected.terminated
    assert actual.truncated == expected.truncated
    a, b = actual.read(), expected.read()
    assert np.array_equal(a.obs["state"], b.obs["state"])
    assert np.array_equal(a.action, b.action)
    assert np.array_equal(a.reward, b.reward)


def test_matches_sequential_reference(backend_name, tmp_path):
    vec_ds = make_dataset(backend_name, tmp_path)
    ref_ds = make_dataset("memory", tmp_path)

    vec = vec_ds.vector_writer()
    vec.reset(obs(0, 100, 200))
    history = [[eid] for eid in vec.episode_ids]

    terminated = {3: [True, False, False]}
    truncated = {5: [False, True, False]}
    for t in range(1, 7):
        vec.step(
            obs(t, 100 + t, 200 + t),
            actions=act(t, 100 + t, 200 + t),
            rewards=rew(t, 100 + t, 200 + t),
            terminated=terminated.get(t),
            truncated=truncated.get(t),
        )
        for i, eid in enumerate(vec.episode_ids):
            if eid != history[i][-1]:
                history[i].append(eid)
    vec.close()

    # env 0 terminates at t=3 and restarts from its t=4 observation; env 1
    # truncates at t=5 and restarts at t=6; env 2 runs through
    expected = [
        [
            write_reference_episode(ref_ds, 0, [1, 2, 3], terminated=True),
            write_reference_episode(ref_ds, 4, [5, 6], end_truncated=True),
        ],
        [
            write_reference_episode(ref_ds, 100, [101, 102, 103, 104, 105], truncated=True),
            write_reference_episode(ref_ds, 106, [], end_truncated=True),
        ],
        [
            write_reference_episode(
                ref_ds, 200, [201, 202, 203, 204, 205, 206], end_truncated=True
            ),
        ],
    ]
    for env, expected_ids in enumerate(expected):
        assert len(history[env]) == len(expected_ids)
        for vec_id, ref_id in zip(history[env], expected_ids):
            assert_episodes_equal(vec_ds.episode(vec_id), ref_ds.episode(ref_id))


def test_staggered_autoreset_writes_reset_rows(backend_name, tmp_path):
    dataset = make_dataset(backend_name, tmp_path)
    vec = dataset.vector_writer()
    vec.reset(obs(0, 100))
    vec.step(obs(1, 101), actions=act(1, 101), rewards=rew(1, 101), terminated=[True, False])
    first = vec.episode_ids[0]
    vec.step(obs(2, 102), actions=act(2, 102), rewards=rew(2, 102))

    assert not dataset.episode(first).ongoing
    fresh = dataset.episode(vec.episode_ids[0])
    # the t=2 obs became the fresh episode's reset row (no transition yet)
    assert fresh.id != first and fresh.ongoing and len(fresh) == 0
    vec.step(obs(3, 103), actions=act(3, 103), rewards=rew(3, 103))
    assert len(fresh) == 1
    row = fresh.step(0)
    assert np.array_equal(row.obs["state"], np.full(3, 2, dtype=np.float32))
    assert np.array_equal(row.action, np.full(2, 3, dtype=np.float32))
    assert row.reward == 3.0
    # env 1 never ended: one ongoing episode of length 3
    assert len(dataset.episode(vec.episode_ids[1])) == 3


def test_first_step_done(backend_name, tmp_path):
    dataset = make_dataset(backend_name, tmp_path)
    vec = dataset.vector_writer()
    vec.reset(obs(0, 100))
    vec.step(obs(1, 101), actions=act(1, 101), rewards=rew(1, 101), terminated=[True, False])
    episode = dataset.episode(vec.episode_ids[0])
    assert episode.terminated and len(episode) == 1
    vec.step(obs(2, 102), actions=act(2, 102), rewards=rew(2, 102))
    assert vec.episode_ids[0] != episode.id


def test_infos_batched(backend_name, tmp_path):
    example = {
        **_example(),
        "initial_info": {"success": False},
        "infos": {"success": np.zeros(2, dtype=np.bool_)},
    }
    schema = DatasetSchema.infer(example)
    path = None if backend_name == "memory" else str(tmp_path / "vec")
    dataset = Dataset.create(schema, path=path, backend=backend_name)

    vec = dataset.vector_writer()
    vec.reset(obs(0, 100), infos={"success": np.array([False, False])})
    vec.step(
        obs(1, 101),
        actions=act(1, 101),
        rewards=rew(1, 101),
        terminated=[False, True],
        infos={"success": np.array([False, True])},
    )
    vec.step(
        obs(2, 102),
        actions=act(2, 102),
        rewards=rew(2, 102),
        infos={"success": np.array([True, False])},
    )
    vec.step(
        obs(3, 103),
        actions=act(3, 103),
        rewards=rew(3, 103),
        infos={"success": np.array([False, True])},
    )
    # env 0: one episode; infos pair with observations, reset info first
    ep0 = dataset.episode(vec.episode_ids[0]).read()
    assert np.array_equal(ep0.info["success"], [False, False, True])
    assert np.array_equal(ep0.next_info["success"], [False, True, False])
    # env 1's second episode starts with the reset-row info from the t=2 call
    fresh = dataset.episode(vec.episode_ids[1]).read()
    assert np.array_equal(fresh.info["success"], [False])
    assert np.array_equal(fresh.next_info["success"], [True])


def test_terminated_and_truncated_same_step(backend_name, tmp_path):
    dataset = make_dataset(backend_name, tmp_path)
    vec = dataset.vector_writer()
    vec.reset(obs(0))
    vec.step(obs(1), actions=act(1), rewards=rew(1), terminated=[True], truncated=[True])
    episode = dataset.episode(vec.episode_ids[0])
    assert episode.terminated and episode.truncated


def test_bare_array_observations():
    example = {
        "initial_observation": np.zeros(3, dtype=np.float32),
        "observations": np.zeros((2, 3), dtype=np.float32),
        "actions": np.zeros((2, 2), dtype=np.float32),
        "rewards": np.zeros(2, dtype=np.float32),
    }
    dataset = Dataset.create(DatasetSchema.infer(example))
    vec = dataset.vector_writer()
    vec.reset(np.ones((2, 3), dtype=np.float32))
    vec.step(
        np.full((2, 3), 2, dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.float32),
        rewards=np.zeros(2, dtype=np.float32),
    )
    data = dataset.episode(vec.episode_ids[0]).read()
    assert np.array_equal(data.obs, [[1, 1, 1]])
    assert np.array_equal(data.next_obs, [[2, 2, 2]])


def test_close_truncates(backend_name, tmp_path):
    dataset = make_dataset(backend_name, tmp_path)
    vec = dataset.vector_writer()
    vec.reset(obs(0, 100))
    vec.step(obs(1, 101), actions=act(1, 101), rewards=rew(1, 101))
    vec.close()
    for episode_id in vec.episode_ids:
        episode = dataset.episode(episode_id)
        assert not episode.ongoing and episode.truncated and not episode.terminated
    vec.close()  # idempotent
    with pytest.raises(ValueError, match="closed"):
        vec.step(obs(2, 102), actions=act(2, 102), rewards=rew(2, 102))


def test_close_no_truncate_leaves_resumable(backend_name, tmp_path):
    dataset = make_dataset(backend_name, tmp_path)
    vec = dataset.vector_writer()
    vec.reset(obs(0, 100))
    vec.close(truncate=False)
    for episode_id in vec.episode_ids:
        assert dataset.episode(episode_id).ongoing
    writer = dataset.resume_episode(vec.episode_ids[0])
    writer.add_step(
        {
            "observations": {"state": np.zeros(3, dtype=np.float32)},
            "actions": np.zeros(2, dtype=np.float32),
            "rewards": 0.0,
        }
    )
    assert len(dataset.episode(vec.episode_ids[0])) == 1


def test_context_manager(backend_name, tmp_path):
    dataset = make_dataset(backend_name, tmp_path)
    with dataset.vector_writer() as vec:
        vec.reset(obs(0, 100))
    for episode_id in vec.episode_ids:
        assert dataset.episode(episode_id).truncated

    with pytest.raises(RuntimeError):
        with dataset.vector_writer() as vec:
            vec.reset(obs(0, 100))
            raise RuntimeError("collection crashed")
    # on exceptions episodes stay ongoing (recoverable), like EpisodeWriter
    for episode_id in vec.episode_ids:
        assert dataset.episode(episode_id).ongoing


def test_reset_midrun_truncates(backend_name, tmp_path):
    dataset = make_dataset(backend_name, tmp_path)
    vec = dataset.vector_writer()
    vec.reset(obs(0, 100))
    first = vec.episode_ids
    vec.step(obs(1, 101), actions=act(1, 101), rewards=rew(1, 101))
    vec.reset(obs(50, 150))
    for episode_id in first:
        episode = dataset.episode(episode_id)
        assert not episode.ongoing and episode.truncated
    assert all(dataset.episode(eid).ongoing for eid in vec.episode_ids)
    vec.step(obs(51, 151), actions=act(51, 151), rewards=rew(51, 151))
    assert np.array_equal(
        dataset.episode(vec.episode_ids[0]).step(0).obs["state"],
        np.full(3, 50, dtype=np.float32),
    )


def test_episode_ids_across_autoresets(backend_name, tmp_path):
    dataset = make_dataset(backend_name, tmp_path)
    vec = dataset.vector_writer(num_envs=2)
    vec.reset(obs(0, 100))
    ids = vec.episode_ids
    vec.step(obs(1, 101), actions=act(1, 101), rewards=rew(1, 101), truncated=[True, False])
    # the done env still points at its just-finalized episode
    assert vec.episode_ids == ids
    vec.step(obs(2, 102), actions=act(2, 102), rewards=rew(2, 102))
    assert vec.episode_ids[0] != ids[0] and vec.episode_ids[1] == ids[1]


def test_errors():
    dataset = Dataset.create(DatasetSchema.infer(_example()))
    vec = dataset.vector_writer()
    with pytest.raises(ValueError, match="reset"):
        vec.step(obs(0), actions=act(0), rewards=rew(0))

    with pytest.raises(ValueError, match="2 envs"):
        dataset.vector_writer(num_envs=2).reset(obs(0, 1, 2))
    with pytest.raises(ValueError, match="positive"):
        dataset.vector_writer(num_envs=0)

    vec.reset(obs(0, 100))
    with pytest.raises(ValueError, match="2 envs"):
        vec.step(obs(1), actions=act(1), rewards=rew(1))
    with pytest.raises(ValueError, match="one length"):
        vec.step(obs(1, 101), actions=act(1), rewards=rew(1, 101))
    with pytest.raises(ValueError, match="terminated"):
        vec.step(obs(1, 101), actions=act(1, 101), rewards=rew(1, 101), terminated=[True])
