import numpy as np
import pytest

from episodata import Dataset, Segment, SpaceView


def test_access_patterns(dataset):
    obs = dataset.episode(0).segment(0, 4)
    # flat dict-style
    assert obs["front_camera"].shape == (4, 3, 8, 8)
    # space attribute access
    assert obs.image.front_camera.shape == (4, 3, 8, 8)
    assert np.array_equal(obs.image["front_camera"], obs["front_camera"])
    # iteration over a space
    keys = dict(obs.image.items())
    assert set(keys) == {"front_camera", "wrist_camera"}
    # stacking valid within a space
    assert obs.image.stacked().shape == (2, 4, 3, 8, 8)


def test_trivial_space_unwraps_to_array(dataset):
    seg = dataset.episode(0).segment(0, 4)
    # the action/reward spaces hold a single same-named field, so attribute
    # access resolves straight to the array
    assert isinstance(seg.action, np.ndarray)
    assert seg.action.shape == (4, 2)
    assert np.array_equal(seg.action, seg["action"])
    assert isinstance(seg.reward, np.ndarray)
    assert seg.reward.shape == (4,)
    # a structural space keeps its view even with differently-named fields
    assert isinstance(seg.image, SpaceView)
    # the view of a trivial space stays reachable explicitly
    view = seg.space_view("action")
    assert isinstance(view, SpaceView)
    assert set(view) == {"action"}


def test_lone_same_named_field_unwraps():
    episode = {
        "initial_observation": {"image": np.zeros((3, 8, 8), dtype=np.uint8)},
        "observations": {"image": np.zeros((4, 3, 8, 8), dtype=np.uint8)},
        "actions": np.zeros((4, 2), dtype=np.float32),
        "rewards": np.zeros(4, dtype=np.float32),
    }
    seg = Dataset.from_episodes([episode]).episode(0).read()
    assert isinstance(seg.image, np.ndarray)
    assert seg.image.shape == (4, 3, 8, 8)


def test_shadowed_space_stays_navigable():
    episode = {
        "initial_observation": {"o": np.zeros(3, dtype=np.float32)},
        "observations": {"o": np.zeros((4, 3), dtype=np.float32)},
        "actions": {
            "action": np.zeros((4, 2), dtype=np.float32),
            "action2": np.ones((4, 2), dtype=np.float32),
        },
    }
    with pytest.warns(UserWarning, match="shadowed by space"):
        ds = Dataset.from_episodes([episode])
    seg = ds.episode(0).read()
    # with siblings the space wins attribute lookup and stays navigable
    assert isinstance(seg.action, SpaceView)
    assert seg.action.action2.shape == (4, 2)
    assert seg.action.action.shape == (4, 2)
    assert set(seg.space_view("action")) == {"action", "action2"}
    # flat item access still reads the shadowed field itself
    assert isinstance(seg["action"], np.ndarray)
    assert np.array_equal(seg["action"], seg.action.action)


def test_unknown_access_raises(dataset):
    obs = dataset.episode(0).segment(0, 2)
    with pytest.raises(AttributeError):
        _ = obs.nonexistent
    with pytest.raises(AttributeError):
        _ = obs.image.nonexistent
    with pytest.raises(KeyError):
        _ = obs["nonexistent"]


def test_field_selection(dataset):
    obs = dataset.episode(0).segment(0, 3, fields=["state", "action"])
    assert set(obs.keys()) == {"state", "action"}
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
    assert set(segment.actions) == {"action"}
    assert set(segment.rewards) == {"reward"}
    assert segment.actions["action"].shape == (4, 2)


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
    assert last["state"].shape == (5,)
    assert last.terminated and not last.truncated and last.mask
    assert not episode.step(0).terminated


def test_role_views_preserve_flags(dataset):
    segment = dataset.episode(0).read()
    rewards = segment.rewards
    assert isinstance(rewards, Segment)
    assert np.array_equal(rewards.terminated, segment.terminated)
    assert np.array_equal(rewards.mask, segment.mask)
