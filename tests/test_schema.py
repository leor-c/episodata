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
        "observations": {
            "small": np.zeros((4, 3, 8, 8), dtype=np.uint8),
            "large": np.zeros((4, 3, 16, 16), dtype=np.uint8),
        }
    }
    schema = DatasetSchema.infer(episode)
    assert schema.field("small").space != schema.field("large").space


def test_action_space_naming():
    # a single action component keeps the plain "action" space
    schema = DatasetSchema.infer(make_episode(5))
    assert schema.field("action").space == "action"
    # structurally distinct components get role-prefixed structural names
    episode = {
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
        "observations": {"state": np.zeros((4, 2), dtype=np.float32)},
        "actions": np.zeros((4, 2), dtype=np.float32),
    }
    schema = DatasetSchema.infer(same_format)
    assert schema.field("state").space != schema.field("action").space


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


def test_rename_space_updates_fields():
    schema = DatasetSchema.infer(make_episode(5))
    schema.rename_space("vector", "proprio")
    assert schema.field("state").space == "proprio"
    assert "vector" not in schema.spaces