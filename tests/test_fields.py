import warnings

import numpy as np
import pytest

from episodata import Dataset, Fields, Segment


def test_access_patterns(dataset):
    seg = dataset.episode(0).segment(0, 4)
    # role → flat dict-style
    assert seg.obs["front_camera"].shape == (4, 3, 8, 8)
    # role → field attribute access
    assert seg.obs.front_camera.shape == (4, 3, 8, 8)
    # iteration over a role
    assert set(seg.obs.keys()) == {"front_camera", "wrist_camera", "state"}


def test_role_aliases_are_identical(dataset):
    seg = dataset.episode(0).segment(0, 4)
    assert seg.obs is seg.observation
    assert seg.observations is seg.observation
    assert seg.actions is seg.action
    assert seg.rewards is seg.reward
    assert seg.infos is seg.info
    assert seg.next_obs is seg.next_observation
    assert seg.next_observations is seg.next_observation
    assert seg.next_infos is seg.next_info


def test_bare_role_unwraps_to_array(dataset):
    seg = dataset.episode(0).segment(0, 4)
    # a role whose only field carries the role's own name resolves straight
    # to the array
    assert isinstance(seg.action, np.ndarray)
    assert seg.action.shape == (4, 2)
    assert isinstance(seg.reward, np.ndarray)
    assert seg.reward.shape == (4,)
    # a multi-field role keeps the view
    assert isinstance(seg.obs, Fields)
    # unwrapping survives field selection
    selected = seg.select(["state", "action"])
    assert isinstance(selected.action, np.ndarray)
    assert selected.action.shape == (4, 2)


def test_bare_observation_unwraps_to_array():
    episode = {
        "initial_observation": np.zeros(3, dtype=np.float32),
        "observations": np.zeros((4, 3), dtype=np.float32),
        "actions": np.zeros((4, 2), dtype=np.float32),
        "rewards": np.zeros(4, dtype=np.float32),
    }
    seg = Dataset.from_episodes([episode]).episode(0).read()
    assert isinstance(seg.obs, np.ndarray)
    assert seg.obs.shape == (4, 3)
    assert isinstance(seg.next_obs, np.ndarray)
    assert seg.next_obs.shape == (4, 3)


def test_multi_field_action_role_stays_a_view():
    episode = {
        "initial_observation": {"o": np.zeros(3, dtype=np.float32)},
        "observations": {"o": np.zeros((4, 3), dtype=np.float32)},
        "actions": {
            "action": np.zeros((4, 2), dtype=np.float32),
            "action2": np.ones((4, 2), dtype=np.float32),
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing shadows anything anymore
        ds = Dataset.from_episodes([episode])
    seg = ds.episode(0).read()
    assert isinstance(seg.action, Fields)
    assert seg.action.action.shape == (4, 2)
    assert seg.action.action2.shape == (4, 2)
    assert np.array_equal(seg.action["action"], seg.action.action)
    assert seg.actions is seg.action


def test_no_top_level_shortcuts(dataset):
    seg = dataset.episode(0).segment(0, 4)
    with pytest.raises(AttributeError):
        _ = seg.front_camera
    with pytest.raises(AttributeError):
        _ = seg.image
    with pytest.raises(TypeError):
        _ = seg["front_camera"]


def test_unknown_access_raises(dataset):
    seg = dataset.episode(0).segment(0, 2)
    with pytest.raises(AttributeError):
        _ = seg.obs.nonexistent
    with pytest.raises(KeyError):
        _ = seg.obs["nonexistent"]


def test_empty_role_is_empty_view(dataset):
    seg = dataset.episode(0).segment(0, 3, fields=["state", "action"])
    reward = seg.reward
    assert isinstance(reward, Fields)
    assert len(reward) == 0


def test_field_selection(dataset):
    seg = dataset.episode(0).segment(0, 3, fields=["state", "action"])
    assert set(seg.obs.keys()) == {"state"}
    assert seg.action.shape == (3, 2)
    with pytest.raises(KeyError):
        dataset.episode(0).segment(0, 3, fields=["missing"])


def test_episode_methods_return_segments(dataset):
    episode = dataset.episode(0)
    segment = episode.segment(0, 4)
    assert isinstance(segment, Segment)
    assert isinstance(episode.read(), Segment)
    assert isinstance(episode.step(0), Segment)


def test_role_views(dataset):
    segment = dataset.episode(0).segment(0, 4)
    assert set(segment.observations) == {"front_camera", "wrist_camera", "state"}
    assert segment.action.shape == (4, 2)
    assert segment.reward.shape == (4,)


def test_segment_flags_only_at_episode_end(dataset):
    episode = dataset.episode(0)  # length 10, terminated
    mid = episode.segment(0, 4)
    assert mid.terminated.shape == (4,)
    assert mid.terminated.sum() == 0
    assert mid.mask.all()
    tail = episode.segment(6)  # reaches the final step
    assert tail.terminated[-1] and tail.terminated.sum() == 1
    assert not tail.truncated.any()
    # episode 1 is neither terminated nor truncated
    full = dataset.episode(1).read()
    assert not full.terminated.any() and not full.truncated.any()


def test_step_has_scalar_flags(dataset):
    episode = dataset.episode(0)
    last = episode.step(-1)
    assert last.obs["state"].shape == (5,)
    assert last.terminated and not last.truncated and last.mask
    assert not episode.step(0).terminated
