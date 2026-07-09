"""normalize_full_episode: the bulk-import shape contract (obs[T+1] vs
actions/rewards[T]) and the reset-row synthesis it performs."""

import numpy as np
import pytest

from episodata.normalize import normalize_full_episode


def test_synthesizes_zero_reset_row():
    episode = {
        "observations": {"x": np.array([[0.0], [1.0], [2.0]], dtype=np.float32)},
        "actions": np.array([[10.0], [20.0]], dtype=np.float32),
        "rewards": np.array([1.0, 2.0], dtype=np.float32),
        "terminated": True,
    }
    normalized = normalize_full_episode(episode)
    assert normalized.length == 3
    assert np.array_equal(normalized.fields["x"], [[0.0], [1.0], [2.0]])
    assert np.array_equal(normalized.fields["action"], [[0.0], [10.0], [20.0]])
    assert np.array_equal(normalized.fields["reward"], [0.0, 1.0, 2.0])
    assert normalized.terminated


def test_step_count_mismatch_raises():
    episode = {
        "observations": {"x": np.zeros((3, 1), dtype=np.float32)},
        "actions": np.zeros((3, 1), dtype=np.float32),  # should be 2, one fewer
        "rewards": np.zeros(3, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="expected 2"):
        normalize_full_episode(episode)


def test_infos_wrong_length_raises():
    episode = {
        "observations": {"x": np.zeros((3, 1), dtype=np.float32)},
        "actions": np.zeros((2, 1), dtype=np.float32),
        "rewards": np.zeros(2, dtype=np.float32),
        "infos": {"success": np.zeros(2, dtype=np.bool_)},  # should be 3, matching observations
    }
    with pytest.raises(ValueError, match="observations/infos"):
        normalize_full_episode(episode)


def test_infos_matches_observations_and_is_never_zero_filled():
    episode = {
        "observations": {"x": np.zeros((3, 1), dtype=np.float32)},
        "actions": np.zeros((2, 1), dtype=np.float32),
        "rewards": np.zeros(2, dtype=np.float32),
        "infos": {"success": np.array([True, False, True])},
    }
    normalized = normalize_full_episode(episode)
    assert np.array_equal(normalized.fields["success"], [True, False, True])


def test_infos_entirely_omitted_is_fine():
    episode = {
        "observations": {"x": np.zeros((3, 1), dtype=np.float32)},
        "actions": np.zeros((2, 1), dtype=np.float32),
        "rewards": np.zeros(2, dtype=np.float32),
    }
    normalized = normalize_full_episode(episode)
    assert "success" not in normalized.fields


def test_reset_only_episode_has_zero_steps():
    # just the reset row, no steps taken
    episode = {
        "observations": {"x": np.zeros((1, 1), dtype=np.float32) + 5},
        "actions": np.zeros((0, 1), dtype=np.float32),
        "rewards": np.zeros(0, dtype=np.float32),
    }
    normalized = normalize_full_episode(episode)
    assert normalized.length == 1
    assert np.array_equal(normalized.fields["action"], [[0.0]])
    assert np.array_equal(normalized.fields["reward"], [0.0])
