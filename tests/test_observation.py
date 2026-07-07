import numpy as np
import pytest


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