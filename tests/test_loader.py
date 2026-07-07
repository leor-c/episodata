import numpy as np
import pytest

from episodata import Dataset
from tests.conftest import make_episode


def test_segment_batch_shapes(dataset):
    loader = dataset.loader(
        fields=["front_camera", "state", "action"],
        sequence_length=4,
        batch_size=6,
        seed=0,
    )
    batch = loader.sample()
    assert batch["front_camera"].shape == (6, 4, 3, 8, 8)
    assert batch.image.front_camera.shape == (6, 4, 3, 8, 8)
    assert batch["state"].shape == (6, 4, 5)
    assert batch.terminated.shape == (6, 4)


def test_context_target_windows(dataset):
    loader = dataset.loader(context_length=2, target_length=3, batch_size=4, seed=0)
    batch = loader.sample()
    assert batch["state"].shape == (4, 5, 5)
    assert batch.context["state"].shape == (4, 2, 5)
    assert batch.target["state"].shape == (4, 3, 5)
    # target follows context contiguously
    assert np.array_equal(
        np.concatenate([batch.context["state"], batch.target["state"]], axis=1),
        batch["state"],
    )


def test_seed_determinism(dataset):
    a = dataset.loader(sequence_length=3, batch_size=5, seed=42).sample()
    b = dataset.loader(sequence_length=3, batch_size=5, seed=42).sample()
    assert np.array_equal(a["state"], b["state"])


def test_sequential_scan_covers_all_windows(dataset):
    loader = dataset.loader(fields=["reward"], sequence_length=4, batch_size=3, shuffle=False)
    windows = [w for batch in loader for w in batch["reward"]]
    # episode lengths 10 and 7 -> (10-4+1) + (7-4+1) = 11 windows
    assert len(windows) == 11
    source = make_episode(10, seed=0)["rewards"]
    assert np.array_equal(windows[0], source[0:4])
    assert np.array_equal(windows[6], source[6:10])


def test_terminated_flag_only_on_final_step(dataset):
    loader = dataset.loader(fields=["reward"], sequence_length=4, batch_size=2, shuffle=False)
    batches = list(loader)
    flat_terminated = np.concatenate([b.terminated for b in batches], axis=0)
    # episode 0 (length 10, terminated) contributes windows 0..6; only the
    # window ending at step 9 carries a True, on its last position.
    assert flat_terminated[:6].sum() == 0
    assert flat_terminated[6, 3] and flat_terminated[6].sum() == 1
    # episode 1 is neither terminated nor truncated
    assert flat_terminated[7:].sum() == 0


def test_transition_alignment_action_in():
    # action-in: row t holds the action/reward that led to obs t, so a
    # transition's action/reward come from row t + 1
    t = np.arange(8, dtype=np.float32)
    episode = {
        "observations": {"x": t[:, None]},
        "actions": (10 * t)[:, None],
        "rewards": 100 * t,
        "terminated": True,
    }
    dataset = Dataset.from_episodes([episode])
    batch = dataset.sample_transitions(batch_size=32, seed=1)
    s_next = batch.next_observations["x"][:, 0]
    assert np.array_equal(s_next, batch.observations["x"][:, 0] + 1)
    assert np.array_equal(batch.actions["action"][:, 0], 10 * s_next)
    assert np.array_equal(batch.rewards, 100 * s_next)


def test_transition_sampling(dataset):
    transitions = dataset.sample_transitions(batch_size=8, seed=0)
    assert transitions.observations["front_camera"].shape == (8, 3, 8, 8)
    assert transitions.next_observations["state"].shape == (8, 5)
    assert transitions.actions["action"].shape == (8, 2)
    assert transitions.rewards.shape == (8,)
    assert transitions.terminated.dtype == bool


def test_filter(dataset):
    loader = dataset.loader(
        fields=["reward"],
        sequence_length=2,
        batch_size=4,
        seed=0,
        filter=lambda ep: len(ep) > 8,
    )
    scan = dataset.loader(
        fields=["reward"], sequence_length=2, shuffle=False, filter=lambda ep: len(ep) > 8
    )
    # only episode 0 (length 10) passes the filter -> 9 windows
    assert sum(len(b["reward"]) for b in scan) == 9
    loader.sample()  # sampling under the filter works


def test_window_too_long_raises(dataset):
    with pytest.raises(ValueError, match="no episode"):
        dataset.loader(sequence_length=100).sample()


def test_online_episodes_become_sampleable(dataset):
    loader = dataset.loader(fields=["reward"], sequence_length=15, batch_size=2, seed=0)
    with pytest.raises(ValueError):
        loader.sample()
    dataset.add_episode(make_episode(20, seed=3))
    batch = loader.sample()
    assert batch["reward"].shape == (2, 15)
