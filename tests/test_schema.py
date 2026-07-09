import warnings

import numpy as np
import pytest

from episodata import DatasetSchema, FieldSpec, SpaceSpec
from tests.conftest import make_episode


def test_infer_groups_structurally_identical_fields():
    schema = DatasetSchema.infer(make_episode(5))
    assert schema.field("front_camera").space == schema.field("wrist_camera").space
    assert schema.field("state").space != schema.field("front_camera").space
    image_space = schema.space_of("front_camera")
    assert image_space.shape == (3, 8, 8)
    assert image_space.dtype == "uint8"
    assert image_space.layout == "CHW"
    assert (image_space.low, image_space.high) == (0, 255)
    assert schema.space_of("state").shape == (5,)
    assert schema.field("action").role == "action"
    assert schema.field("reward").role == "reward"


def test_infer_separates_same_base_different_spec():
    episode = {
        "initial_observation": {
            "small": np.zeros((3, 8, 8), dtype=np.uint8),
            "large": np.zeros((3, 16, 16), dtype=np.uint8),
        },
        "observations": {
            "small": np.zeros((4, 3, 8, 8), dtype=np.uint8),
            "large": np.zeros((4, 3, 16, 16), dtype=np.uint8),
        },
    }
    schema = DatasetSchema.infer(episode)
    assert schema.field("small").space != schema.field("large").space


def test_action_space_naming():
    # a single action component keeps the plain "action" space
    schema = DatasetSchema.infer(make_episode(5))
    assert schema.field("action").space == "action"
    # structurally distinct components get role-prefixed structural names
    episode = {
        "initial_observation": {"o": np.zeros(3, dtype=np.float32)},
        "observations": {"o": np.zeros((4, 3), dtype=np.float32)},
        "actions": {
            "camera": np.zeros((4, 2), dtype=np.float32),
            "jump": np.zeros(4, dtype=np.uint8),
        },
    }
    schema = DatasetSchema.infer(episode)
    assert schema.field("camera").space == "action_vector"
    assert schema.field("jump").space == "action_scalar"
    # actions never merge into observation spaces, even with matching format
    same_format = {
        "initial_observation": {"state": np.zeros(2, dtype=np.float32)},
        "observations": {"state": np.zeros((4, 2), dtype=np.float32)},
        "actions": np.zeros((4, 2), dtype=np.float32),
    }
    schema = DatasetSchema.infer(same_format)
    assert schema.field("state").space != schema.field("action").space


def test_info_never_merges_into_observation_space():
    # obs and info fields with identical formats stay in role-bound spaces
    episode = {
        "initial_observation": {"pos": np.zeros(3, dtype=np.float32)},
        "observations": {"pos": np.zeros((4, 3), dtype=np.float32)},
        "initial_info": {"debug_vec": np.zeros(3, dtype=np.float32)},
        "infos": {"debug_vec": np.zeros((4, 3), dtype=np.float32)},
    }
    schema = DatasetSchema.infer(episode)
    assert schema.field("pos").space == "vector"
    assert schema.field("debug_vec").space == "info_vector"
    assert schema.space("vector").role == "observation"
    assert schema.space("info_vector").role == "info"
    assert schema.fields_in_space("vector") == ["pos"]


def test_same_named_field_and_space_never_warn():
    # role-first access with explicit .space() leaves nothing to shadow: a
    # field named after its space is unambiguous, siblings or not
    episode = {
        "initial_observation": {"o": np.zeros(3, dtype=np.float32)},
        "observations": {"o": np.zeros((4, 3), dtype=np.float32)},
        "actions": {
            "action": np.zeros((4, 2), dtype=np.float32),
            "action2": np.zeros((4, 2), dtype=np.float32),
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DatasetSchema.infer(episode)
        DatasetSchema.infer(make_episode(5))


def test_json_roundtrip():
    schema = DatasetSchema.infer(make_episode(5))
    restored = DatasetSchema.from_json(schema.to_json())
    assert restored.to_dict() == schema.to_dict()


def test_declared_schema_validates_references():
    with pytest.raises(ValueError, match="unknown space"):
        DatasetSchema(
            spaces=[SpaceSpec(key="image", shape=(3, 8, 8), dtype="uint8")],
            fields=[FieldSpec(key="cam", space="nope")],
        )


def test_declared_schema_rejects_role_mismatch():
    with pytest.raises(ValueError, match="role"):
        DatasetSchema(
            spaces=[SpaceSpec(key="vector", shape=(3,), dtype="float32")],
            fields=[FieldSpec(key="debug", space="vector", role="info")],
        )


def test_space_role_roundtrips_and_defaults():
    schema = DatasetSchema.infer(make_episode(5))
    restored = DatasetSchema.from_json(schema.to_json())
    assert restored.space("action").role == "action"
    # schemas persisted before spaces carried a role default to observation
    assert SpaceSpec.from_dict({"key": "v", "shape": [3], "dtype": "float32"}).role == "observation"


def test_rename_space_updates_fields():
    schema = DatasetSchema.infer(make_episode(5))
    schema.rename_space("vector", "proprio")
    assert schema.field("state").space == "proprio"
    assert "vector" not in schema.spaces