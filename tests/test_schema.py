import numpy as np
import pytest

from episodata import DatasetSchema, FieldSpec
from tests.conftest import make_episode


def test_infer_declares_per_field_format():
    schema = DatasetSchema.infer(make_episode(5))
    front = schema.field("front_camera")
    assert front.shape == (3, 8, 8)
    assert front.dtype == "uint8"
    assert front.layout == "CHW"
    assert (front.low, front.high) == (0, 255)
    assert schema.field("wrist_camera").shape == (3, 8, 8)
    assert schema.field("state").shape == (5,)
    assert schema.field("state").layout is None
    assert schema.field("action").role == "action"
    assert schema.field("reward").role == "reward"


def test_infer_image_layouts():
    episode = {
        "initial_observation": {
            "chw": np.zeros((3, 8, 8), dtype=np.uint8),
            "hwc": np.zeros((8, 8, 3), dtype=np.uint8),
            "gray": np.zeros((8, 8), dtype=np.uint8),
        },
        "observations": {
            "chw": np.zeros((4, 3, 8, 8), dtype=np.uint8),
            "hwc": np.zeros((4, 8, 8, 3), dtype=np.uint8),
            "gray": np.zeros((4, 8, 8), dtype=np.uint8),
        },
    }
    schema = DatasetSchema.infer(episode)
    assert schema.field("chw").layout == "CHW"
    assert schema.field("hwc").layout == "HWC"
    assert schema.field("gray").layout is None
    assert (schema.field("gray").low, schema.field("gray").high) == (0, 255)


def test_infer_roles_and_optionality():
    episode = {
        "initial_observation": {"pos": np.zeros(3, dtype=np.float32)},
        "observations": {"pos": np.zeros((4, 3), dtype=np.float32)},
        "actions": np.zeros((4, 2), dtype=np.float32),
        "initial_info": {"debug_vec": np.zeros(3, dtype=np.float32)},
        "infos": {"debug_vec": np.zeros((4, 3), dtype=np.float32)},
    }
    schema = DatasetSchema.infer(episode)
    assert schema.field("pos").role == "observation"
    assert schema.field("action").role == "action"
    assert schema.field("debug_vec").role == "info"
    assert schema.field("debug_vec").optional
    assert not schema.field("pos").optional
    assert schema.field_keys("observation") == ["pos"]


def test_json_roundtrip():
    schema = DatasetSchema.infer(make_episode(5))
    restored = DatasetSchema.from_json(schema.to_json())
    assert restored.to_dict() == schema.to_dict()


def test_declared_schema_rejects_duplicate_keys():
    spec = FieldSpec(key="cam", shape=(3, 8, 8), dtype="uint8")
    with pytest.raises(ValueError, match="duplicate field key"):
        DatasetSchema(fields=[spec, spec])


def test_field_spec_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown role"):
        FieldSpec(key="x", shape=(3,), dtype="float32", role="nope")


def test_from_dict_rejects_other_schema_versions():
    d = DatasetSchema.infer(make_episode(5)).to_dict()
    d["schema_version"] = 1
    with pytest.raises(ValueError, match="unsupported schema version"):
        DatasetSchema.from_dict(d)


def test_rename_field():
    schema = DatasetSchema.infer(make_episode(5))
    schema.rename_field("state", "proprio")
    assert "state" not in schema.fields
    assert schema.field("proprio").shape == (5,)
    assert schema.field("proprio").key == "proprio"
    with pytest.raises(ValueError, match="already exists"):
        schema.rename_field("proprio", "action")


def test_resolve_fields():
    schema = DatasetSchema.infer(make_episode(5))
    assert schema.resolve_fields(None) == list(schema.fields)
    assert schema.resolve_fields(["state"]) == ["state"]
    with pytest.raises(KeyError, match="unknown field"):
        schema.resolve_fields(["nope"])


def test_resolve_fields_expands_path_prefix():
    schema = DatasetSchema(
        fields=[
            FieldSpec(key="keyboard/w", shape=(), dtype="bool"),
            FieldSpec(key="keyboard/s", shape=(), dtype="bool"),
            FieldSpec(key="mouse", shape=(2,), dtype="float32"),
        ]
    )
    assert schema.resolve_fields(["keyboard"]) == ["keyboard/w", "keyboard/s"]
    assert schema.resolve_fields(["mouse", "keyboard/w"]) == ["mouse", "keyboard/w"]


def test_validate_fields():
    schema = DatasetSchema.infer(make_episode(5))
    fields = {
        key: np.zeros((3, *spec.shape), dtype=spec.dtype)
        for key, spec in schema.fields.items()
    }
    out = schema.validate_fields(dict(fields, reward=np.zeros(3, dtype=np.float64)))
    assert out["reward"].dtype == schema.field("reward").dtype  # cast to logical dtype

    with pytest.raises(KeyError, match="not in the schema"):
        schema.validate_fields(dict(fields, extra=np.zeros(3)))
    with pytest.raises(ValueError, match="expected"):
        schema.validate_fields(dict(fields, state=np.zeros((3, 6), dtype=np.float32)))
    with pytest.raises(ValueError, match="missing required fields"):
        schema.validate_fields({"state": fields["state"]})
