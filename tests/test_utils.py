import numpy as np
import pytest

from episodata import DatasetSchema
from tests.conftest import requires_gym

pytest.importorskip("gymnasium")
import gymnasium.spaces as spaces  # noqa: E402

from episodata.utils import schema_from_gym_spaces  # noqa: E402

pytestmark = requires_gym


def test_box_and_discrete_cartpole_style():
    schema = schema_from_gym_spaces(
        spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        spaces.Discrete(2),
    )
    assert isinstance(schema, DatasetSchema)
    obs = schema.field("observation")
    assert obs.shape == (4,)
    assert obs.dtype == "float32"
    assert obs.role == "observation"
    assert (obs.low, obs.high) is not None  # uniform (-inf, inf) is still uniform
    assert obs.low == float("-inf") and obs.high == float("inf")

    action = schema.field("action")
    assert action.shape == ()
    assert action.dtype == "int64"
    assert action.role == "action"
    assert (action.low, action.high) == (0, 1)

    reward = schema.field("reward")
    assert reward.shape == ()
    assert reward.dtype == "float32"
    assert reward.role == "reward"


def test_discrete_start_offset():
    schema = schema_from_gym_spaces(
        spaces.Box(0, 1, shape=(2,), dtype=np.float32),
        spaces.Discrete(3, start=2),
    )
    action = schema.field("action")
    assert (action.low, action.high) == (2, 4)


def test_dict_observation_flattens_with_path_keys():
    obs_space = spaces.Dict({
        "front_camera": spaces.Box(0, 255, shape=(3, 8, 8), dtype=np.uint8),
        "state": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
    })
    schema = schema_from_gym_spaces(obs_space, spaces.Discrete(2))
    assert schema.field("front_camera").shape == (3, 8, 8)
    assert schema.field("front_camera").layout == "CHW"
    assert (schema.field("front_camera").low, schema.field("front_camera").high) == (0, 255)
    assert schema.field("state").shape == (5,)
    assert (schema.field("state").low, schema.field("state").high) == (-1.0, 1.0)


def test_nested_dict_uses_sep_path_keys():
    obs_space = spaces.Dict({
        "keyboard": spaces.Dict({
            "w": spaces.Discrete(2),
            "s": spaces.Discrete(2),
        }),
    })
    schema = schema_from_gym_spaces(obs_space, spaces.Discrete(2))
    assert schema.field("keyboard/w").role == "observation"
    assert schema.field("keyboard/s").role == "observation"
    assert sorted(schema.resolve_fields(["keyboard"])) == ["keyboard/s", "keyboard/w"]


def test_multidiscrete_and_multibinary():
    obs_space = spaces.Dict({
        "counts": spaces.MultiDiscrete([3, 4]),
        "flags": spaces.MultiBinary(5),
    })
    schema = schema_from_gym_spaces(obs_space, spaces.Discrete(2))
    counts = schema.field("counts")
    assert counts.shape == (2,)
    assert counts.dtype == "int64"
    assert (counts.low, counts.high) is not None
    assert counts.low == 0
    assert counts.high is None  # nvec entries differ (3 vs 4): not uniform

    flags = schema.field("flags")
    assert flags.shape == (5,)
    assert (flags.low, flags.high) == (0, 1)


def test_non_uniform_box_bounds_are_none():
    obs_space = spaces.Box(
        low=np.array([-1.0, -2.0], dtype=np.float32),
        high=np.array([1.0, 2.0], dtype=np.float32),
        dtype=np.float32,
    )
    schema = schema_from_gym_spaces(obs_space, spaces.Discrete(2))
    obs = schema.field("observation")
    assert obs.low is None
    assert obs.high is None


def test_tuple_uses_positional_keys():
    action_space = spaces.Tuple([spaces.Discrete(2), spaces.Box(0, 1, shape=(3,), dtype=np.float32)])
    schema = schema_from_gym_spaces(spaces.Box(0, 1, shape=(4,), dtype=np.float32), action_space)
    assert schema.field("0").role == "action"
    assert schema.field("0").shape == ()
    assert schema.field("1").shape == (3,)


def test_unsupported_space_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Text"):
        schema_from_gym_spaces(spaces.Text(10), spaces.Discrete(2))


def test_missing_gymnasium_raises_clear_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gymnasium.spaces" or name.startswith("gymnasium"):
            raise ImportError("no gymnasium")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="episodata\\[gym\\]"):
        schema_from_gym_spaces(spaces.Discrete(2), spaces.Discrete(2))
