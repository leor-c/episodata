"""The transition-view read contract: round-trip identities against the
write-side terms, zero-copy views, and the boundary cases."""

import numpy as np

from episodata import Dataset
from tests.conftest import make_episode


def test_round_trip_identities(dataset):
    source = make_episode(10, seed=0)
    seg = dataset.episode(0).read()
    # actions/rewards come back exactly as written
    assert np.array_equal(seg.action, source["actions"]["action"])
    assert np.array_equal(seg.reward, source["rewards"])
    # next_observation is exactly the written per-step observations
    for key, expected in source["observations"].items():
        assert np.array_equal(seg.next_obs[key], expected)
        # observations are the same sequence shifted: reset obs first
        assert np.array_equal(seg.obs[key][0], source["initial_observation"][key])
        assert np.array_equal(seg.obs[key][1:], expected[:-1])


def test_views_share_one_row_buffer(dataset):
    seg = dataset.episode(0).segment(0, 6)
    assert np.shares_memory(seg.obs["state"], seg.next_obs["state"])
    assert np.shares_memory(seg.obs["state"], seg._rows["state"])
    assert np.shares_memory(seg.reward, seg._rows["reward"])


def test_collated_batch_shares_one_row_buffer(dataset):
    segments = dataset.segments(fields=["state", "action"], sequence_length=4)
    batch = segments.collate([segments[0], segments[1]])
    assert np.shares_memory(batch.obs["state"], batch.next_obs["state"])
    assert np.shares_memory(batch.obs["state"], batch._rows["state"])
    assert batch.obs["state"].shape == (2, 4, 5)
    assert batch.next_obs["state"].shape == (2, 4, 5)


def test_select_preserves_transition_views(dataset):
    seg = dataset.episode(0).segment(0, 4)
    selected = seg.select(["state", "reward"])
    assert set(selected.obs) == {"state"}
    assert np.array_equal(selected.obs["state"], seg.obs["state"])
    assert np.array_equal(selected.reward, seg.reward)
    assert np.array_equal(selected.next_obs["state"], seg.next_obs["state"])
    assert np.array_equal(selected.mask, seg.mask)


def test_next_info_pairs_with_next_observation():
    episode = {
        "initial_observation": {"x": np.zeros(1, dtype=np.float32)},
        "initial_info": {"success": np.array(False)},
        "observations": {"x": np.ones((2, 1), dtype=np.float32)},
        "actions": np.ones((2, 1), dtype=np.float32),
        "rewards": np.ones(2, dtype=np.float32),
        "infos": {"success": np.array([False, True])},
    }
    seg = Dataset.from_episodes([episode]).episode(0).read()
    assert np.array_equal(seg.info["success"], [False, False])
    assert np.array_equal(seg.next_info["success"], [False, True])


def test_reset_only_episode_reads_empty(dataset):
    writer = dataset.new_episode(make_episode(1, seed=5)["initial_observation"])
    episode = dataset.episode(writer.episode_id)
    assert len(episode) == 0
    seg = episode.read()
    assert seg.obs["state"].shape == (0, 5)
    assert seg.next_obs["state"].shape == (0, 5)
    assert seg.terminated.shape == (0,)


def test_bare_episode_reads_empty(dataset):
    # no reset row written yet: nothing to read, length 0
    writer = dataset.new_episode()
    episode = dataset.episode(writer.episode_id)
    assert len(episode) == 0
    seg = episode.read()
    assert seg.obs["state"].shape == (0, 5)
    assert seg.action.shape == (0, 2)
