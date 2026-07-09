"""Action-out adapters: bulk import shift and the streaming writer."""

import numpy as np

from episodata import ActionOutWriter, Dataset, DatasetSchema


def action_out_episode(length: int = 6) -> dict:
    # obs value == t; action 10t and reward 100t are taken AT obs t
    t = np.arange(length, dtype=np.float32)
    return {
        "observations": {"x": t[:, None]},
        "actions": (10 * t)[:, None],
        "rewards": 100 * t,
        "terminated": True,
    }


def test_bulk_import_shifts_alignment():
    dataset = Dataset.from_episodes([action_out_episode()], alignment="action_out")
    data = dataset.episode(0).read()
    # the transition view recovers exactly the source pairing: the action
    # taken AT obs t comes back alongside obs t. The source's final
    # (unusable) action/reward are dropped, so 6 rows make 5 transitions.
    assert np.array_equal(data["x"][:, 0], np.arange(5))
    assert np.array_equal(data.next_observations["x"][:, 0], np.arange(1, 6))
    assert np.array_equal(data["action"][:, 0], [0, 10, 20, 30, 40])
    assert np.array_equal(data["reward"], [0, 100, 200, 300, 400])


def test_transitions_recover_action_out_semantics():
    dataset = Dataset.from_episodes([action_out_episode()], alignment="action_out")
    t = dataset.sample_transitions(batch_size=64, seed=0)
    s = t.observations["x"][:, 0]
    assert np.array_equal(t.next_observations["x"][:, 0], s + 1)
    # the action/reward taken AT s come back paired with s, as the source meant
    assert np.array_equal(t.actions["action"][:, 0], 10 * s)
    assert np.array_equal(t.rewards, 100 * s)


def _stream(writer: ActionOutWriter, steps: int) -> None:
    for t in range(steps):
        writer.add_step(
            {
                "observations": {"x": np.array([float(t)], dtype=np.float32)},
                "actions": np.array([10.0 * t], dtype=np.float32),
                "rewards": 100.0 * t,
            }
        )


def test_action_out_writer_with_final_observation():
    dataset = Dataset.create(DatasetSchema.infer(action_out_episode(), alignment="action_out"))
    writer = ActionOutWriter(dataset.new_episode())
    _stream(writer, 4)
    episode = writer.end(
        terminated=True, final_observation={"x": np.array([4.0], dtype=np.float32)}
    )
    assert len(episode) == 4 and episode.terminated
    data = episode.read()
    assert np.array_equal(data["x"][:, 0], [0, 1, 2, 3])
    assert np.array_equal(data.next_observations["x"][:, 0], [1, 2, 3, 4])
    assert np.array_equal(data["action"][:, 0], [0, 10, 20, 30])
    assert np.array_equal(data["reward"], [0, 100, 200, 300])


def test_action_out_writer_drops_pending_without_final_observation():
    dataset = Dataset.create(DatasetSchema.infer(action_out_episode(), alignment="action_out"))
    writer = ActionOutWriter(dataset.new_episode())
    _stream(writer, 4)
    episode = writer.end(truncated=True)
    assert len(episode) == 3
    data = episode.read()
    assert np.array_equal(data["action"][:, 0], [0, 10, 20])


def test_action_out_step_signals():
    dataset = Dataset.create(DatasetSchema.infer(action_out_episode(), alignment="action_out"))
    writer = ActionOutWriter(dataset.new_episode())
    _stream(writer, 3)
    # terminal observation arrives with its Gymnasium-style signal; no action
    writer.add_step(
        {"observations": {"x": np.array([3.0], dtype=np.float32)}, "terminated": True}
    )
    episode = dataset.episode(writer.episode_id)
    assert not episode.ongoing and episode.terminated
    data = episode.read()
    assert np.array_equal(data["x"][:, 0], [0, 1, 2])
    assert np.array_equal(data.next_observations["x"][:, 0], [1, 2, 3])
    assert np.array_equal(data["action"][:, 0], [0, 10, 20])


def test_writer_and_bulk_import_agree():
    imported = Dataset.from_episodes([action_out_episode(5)], alignment="action_out")
    streamed = Dataset.create(DatasetSchema.infer(action_out_episode(), alignment="action_out"))
    writer = ActionOutWriter(streamed.new_episode())
    _stream(writer, 5)
    writer.end(terminated=True)
    a = imported.episode(0).read()
    b = streamed.episode(0).read()
    for key in a:
        assert np.array_equal(a[key], b[key]), key


def test_writer_and_bulk_import_agree_with_final_observation():
    final = {"x": np.array([5.0], dtype=np.float32)}
    episode = {**action_out_episode(5), "final_observation": final}
    imported = Dataset.from_episodes([episode], alignment="action_out")
    streamed = Dataset.create(DatasetSchema.infer(episode, alignment="action_out"))
    writer = ActionOutWriter(streamed.new_episode())
    _stream(writer, 5)
    writer.end(terminated=True, final_observation=final)
    # with the final observation, nothing is dropped: all 5 actions survive
    assert len(imported.episode(0)) == len(streamed.episode(0)) == 5
    a = imported.episode(0).read()
    b = streamed.episode(0).read()
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    assert np.array_equal(a.next_observations["x"], b.next_observations["x"])
    assert np.array_equal(a["action"][:, 0], [0, 10, 20, 30, 40])
