import numpy as np
import pytest

from episodata.schema import DatasetSchema, FieldSpec
from episodata.segment import Batch, Segment
from tests.conftest import requires_tensordict

pytestmark = requires_tensordict

pytest.importorskip("torch")
pytest.importorskip("tensordict")
import torch  # noqa: E402

from episodata.utils import batch_to_tensordict  # noqa: E402

B, LP1 = 2, 4  # batch of 2, window of L=3 transitions backed by L+1=4 rows


def _make_batch():
    schema = DatasetSchema(
        fields=[
            FieldSpec(key="cam1", shape=(4, 4, 3), dtype="uint8", role="observation"),
            FieldSpec(key="keyboard/w", shape=(), dtype="float32", role="action"),
            FieldSpec(key="keyboard/a", shape=(), dtype="float32", role="action"),
            FieldSpec(key="reward", shape=(), dtype="float32", role="reward"),
        ]
    )
    rng = np.random.default_rng(0)
    rows = {
        "cam1": rng.integers(0, 256, size=(B, LP1, 4, 4, 3), dtype=np.uint8),
        "keyboard/w": rng.standard_normal((B, LP1)).astype(np.float32),
        "keyboard/a": rng.standard_normal((B, LP1)).astype(np.float32),
        "reward": rng.standard_normal((B, LP1)).astype(np.float32),
    }
    flags = dict(
        terminated=np.zeros((B, LP1 - 1), dtype=bool),
        truncated=np.zeros((B, LP1 - 1), dtype=bool),
        mask=np.ones((B, LP1 - 1), dtype=bool),
    )
    return Batch(rows, schema, **flags)


def test_roles_and_grouped_action_nesting():
    td = batch_to_tensordict(_make_batch())
    assert set(td.keys()) == {
        "observation", "action", "reward", "next_observation",
        "terminated", "truncated", "mask",
    }
    assert set(td["action"].keys()) == {"keyboard"}
    assert set(td["action"]["keyboard"].keys()) == {"w", "a"}
    assert td.batch_size == torch.Size([B, LP1 - 1])
    assert td["observation"]["cam1"].shape == (B, LP1 - 1, 4, 4, 3)


def test_observation_and_next_observation_share_storage():
    td = batch_to_tensordict(_make_batch())
    obs = td["observation"]["cam1"]
    next_obs = td["next_observation"]["cam1"]
    assert obs.untyped_storage().data_ptr() == next_obs.untyped_storage().data_ptr()


def test_include_all_observations_shrinks_batch_size_and_stays_zero_copy():
    td = batch_to_tensordict(_make_batch(), include_all_observations=True)
    assert td.batch_size == torch.Size([B])
    all_obs = td["all_observations"]["cam1"]
    assert all_obs.shape == (B, LP1, 4, 4, 3)
    # per-key shapes keep their own (unequal) time length even though
    # batch_size no longer covers it
    assert td["observation"]["cam1"].shape == (B, LP1 - 1, 4, 4, 3)
    assert td["mask"].shape == (B, LP1 - 1)
    assert (
        all_obs.untyped_storage().data_ptr()
        == td["observation"]["cam1"].untyped_storage().data_ptr()
        == td["next_observation"]["cam1"].untyped_storage().data_ptr()
    )


def test_unbatched_segment_include_all_observations_has_empty_batch_size():
    schema = DatasetSchema(
        fields=[
            FieldSpec(key="cam1", shape=(4, 4, 3), dtype="uint8", role="observation"),
            FieldSpec(key="action", shape=(), dtype="float32", role="action"),
            FieldSpec(key="reward", shape=(), dtype="float32", role="reward"),
        ]
    )
    rng = np.random.default_rng(0)
    rows = {
        "cam1": rng.integers(0, 256, size=(LP1, 4, 4, 3), dtype=np.uint8),
        "action": rng.standard_normal(LP1).astype(np.float32),
        "reward": rng.standard_normal(LP1).astype(np.float32),
    }
    seg = Segment(
        rows, schema,
        terminated=np.zeros(LP1 - 1, dtype=bool),
        truncated=np.zeros(LP1 - 1, dtype=bool),
        mask=np.ones(LP1 - 1, dtype=bool),
    )
    td = batch_to_tensordict(seg, include_all_observations=True)
    assert td.batch_size == torch.Size([])
    assert td["all_observations"]["cam1"].shape == (LP1, 4, 4, 3)


def test_device_is_applied_inside_map_not_on_result():
    calls = []
    real_to = torch.Tensor.to

    def counting_to(self, *args, **kwargs):
        calls.append(1)
        return real_to(self, *args, **kwargs)

    torch.Tensor.to = counting_to
    try:
        batch_to_tensordict(_make_batch(), device="cpu")
    finally:
        torch.Tensor.to = real_to
    # 4 fields (cam1, keyboard/w, keyboard/a, reward) + terminated/truncated/mask
    assert len(calls) == 7


def test_empty_role_is_dropped_not_included_as_empty_dict():
    td = batch_to_tensordict(_make_batch())
    assert "info" not in td
    assert "next_info" not in td


def test_missing_flags_raises_clear_error():
    schema = DatasetSchema(
        fields=[
            FieldSpec(key="observation", shape=(), dtype="float32", role="observation"),
            FieldSpec(key="action", shape=(), dtype="float32", role="action"),
            FieldSpec(key="reward", shape=(), dtype="float32", role="reward"),
        ]
    )
    rows = {
        "observation": np.zeros((B, LP1), dtype=np.float32),
        "action": np.zeros((B, LP1), dtype=np.float32),
        "reward": np.zeros((B, LP1), dtype=np.float32),
    }
    batch = Batch(rows, schema)
    with pytest.raises(ValueError, match="terminated/truncated/mask"):
        batch_to_tensordict(batch)


def test_alignment_action_in_gives_uniform_batch_size_and_zero_first_row():
    batch = _make_batch()
    td = batch_to_tensordict(batch, alignment="action_in")
    assert set(td.keys()) == {"all_observations", "action", "reward", "terminated", "truncated", "mask"}
    assert td.batch_size == torch.Size([B, LP1])
    assert td["all_observations"]["cam1"].shape == (B, LP1, 4, 4, 3)
    assert td["mask"].shape == (B, LP1)
    reward = td["reward"]
    assert torch.all(reward[:, 0] == 0)
    assert torch.equal(reward[:, 1:], torch.as_tensor(batch.reward))


def test_alignment_action_out_gives_uniform_batch_size_and_zero_last_row():
    batch = _make_batch()
    td = batch_to_tensordict(batch, alignment="action_out")
    assert td.batch_size == torch.Size([B, LP1])
    reward = td["reward"]
    assert torch.all(reward[:, -1] == 0)
    assert torch.equal(reward[:, :-1], torch.as_tensor(batch.reward))


def test_alignment_drops_role_length_observation_entries_in_favor_of_all_observations():
    td = batch_to_tensordict(_make_batch(), alignment="action_out")
    assert "observation" not in td
    assert "next_observation" not in td


def test_alignment_conflicts_with_include_all_observations():
    with pytest.raises(ValueError, match="include_all_observations"):
        batch_to_tensordict(_make_batch(), alignment="action_in", include_all_observations=True)


def test_alignment_rejects_invalid_value():
    with pytest.raises(ValueError, match="alignment must be"):
        batch_to_tensordict(_make_batch(), alignment="sideways")


def test_missing_torch_raises_clear_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="torch"):
        batch_to_tensordict(_make_batch())
