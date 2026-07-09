"""Bulk-import normalization: the explicit initial/final boundary-row keys
and the reset-row synthesis they drive."""

import numpy as np
import pytest

from episodata.normalize import normalize_action_out_episode, normalize_full_episode


def action_in_episode(**extra):
    episode = {
        "initial_observation": {"x": np.array([0.0], dtype=np.float32)},
        "observations": {"x": np.array([[1.0], [2.0]], dtype=np.float32)},
        "actions": np.array([[10.0], [20.0]], dtype=np.float32),
        "rewards": np.array([1.0, 2.0], dtype=np.float32),
        "terminated": True,
    }
    episode.update(extra)
    return episode


def test_assembles_reset_row_with_zero_action_reward():
    normalized = normalize_full_episode(action_in_episode())
    assert normalized.length == 3
    assert np.array_equal(normalized.fields["x"], [[0.0], [1.0], [2.0]])
    assert np.array_equal(normalized.fields["action"], [[0.0], [10.0], [20.0]])
    assert np.array_equal(normalized.fields["reward"], [0.0, 1.0, 2.0])
    assert normalized.terminated


def test_missing_initial_observation_raises():
    episode = action_in_episode()
    del episode["initial_observation"]
    with pytest.raises(ValueError, match="initial_observation"):
        normalize_full_episode(episode)


def test_initial_observation_field_mismatch_raises():
    episode = action_in_episode(initial_observation={"y": np.array([0.0], dtype=np.float32)})
    with pytest.raises(ValueError, match="must cover exactly the same fields"):
        normalize_full_episode(episode)


def test_infos_without_initial_info_raises():
    episode = action_in_episode(infos={"success": np.array([False, True])})
    with pytest.raises(ValueError, match="must cover exactly the same fields"):
        normalize_full_episode(episode)


def test_initial_info_without_infos_raises():
    episode = action_in_episode(initial_info={"success": np.array(False)})
    with pytest.raises(ValueError, match="must cover exactly the same fields"):
        normalize_full_episode(episode)


def test_paired_infos_are_never_zero_filled():
    episode = action_in_episode(
        infos={"success": np.array([False, True])},
        initial_info={"success": np.array(True)},
    )
    normalized = normalize_full_episode(episode)
    assert np.array_equal(normalized.fields["success"], [True, False, True])


def test_infos_entirely_omitted_is_fine():
    normalized = normalize_full_episode(action_in_episode())
    assert "success" not in normalized.fields


def test_reset_only_episode_has_zero_steps():
    episode = {
        "initial_observation": {"x": np.array([5.0], dtype=np.float32)},
        "observations": {"x": np.zeros((0, 1), dtype=np.float32)},
        "actions": np.zeros((0, 1), dtype=np.float32),
        "rewards": np.zeros(0, dtype=np.float32),
    }
    normalized = normalize_full_episode(episode)
    assert normalized.length == 1
    assert np.array_equal(normalized.fields["x"], [[5.0]])
    assert np.array_equal(normalized.fields["action"], [[0.0]])
    assert np.array_equal(normalized.fields["reward"], [0.0])


def action_out_episode(**extra):
    episode = {
        "observations": {"x": np.array([[0.0], [1.0], [2.0]], dtype=np.float32)},
        "actions": np.array([[10.0], [20.0], [30.0]], dtype=np.float32),
        "rewards": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "terminated": True,
    }
    episode.update(extra)
    return episode


def test_action_out_without_final_observation_drops_last_action():
    normalized = normalize_action_out_episode(action_out_episode())
    assert normalized.length == 3
    assert np.array_equal(normalized.fields["x"], [[0.0], [1.0], [2.0]])
    assert np.array_equal(normalized.fields["action"], [[0.0], [10.0], [20.0]])
    assert np.array_equal(normalized.fields["reward"], [0.0, 1.0, 2.0])


def test_action_out_with_final_observation_keeps_all_actions():
    episode = action_out_episode(final_observation={"x": np.array([3.0], dtype=np.float32)})
    normalized = normalize_action_out_episode(episode)
    assert normalized.length == 4
    assert np.array_equal(normalized.fields["x"], [[0.0], [1.0], [2.0], [3.0]])
    assert np.array_equal(normalized.fields["action"], [[0.0], [10.0], [20.0], [30.0]])
    assert np.array_equal(normalized.fields["reward"], [0.0, 1.0, 2.0, 3.0])


def test_final_info_requires_final_observation():
    episode = action_out_episode(final_info={"success": np.array(True)})
    with pytest.raises(ValueError, match="final_observation"):
        normalize_action_out_episode(episode)


def test_action_out_final_pairing_is_all_or_nothing():
    episode = action_out_episode(
        infos={"success": np.array([False, False, True])},
        final_observation={"x": np.array([3.0], dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="must cover exactly the same fields"):
        normalize_action_out_episode(episode)
    episode["final_info"] = {"success": np.array(True)}
    normalized = normalize_action_out_episode(episode)
    assert np.array_equal(normalized.fields["success"], [False, False, True, True])
