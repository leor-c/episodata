"""Zarr-backend specifics: persistence layout, recovery, and migration.

The generic storage contract is covered by the backend-parametrized suite;
these tests exercise what is unique to the chunked single-store layout.
"""

import os

import numpy as np
import pytest

from episodata import Dataset, DatasetSchema, FieldSpec, Selection, SpaceSpec
from tests.conftest import make_episode, make_steps

pytest.importorskip("zarr")


def make_dataset(path, lengths=(10, 7), **backend_options):
    episodes = [
        make_episode(length, seed=i, terminated=(i == 0))
        for i, length in enumerate(lengths)
    ]
    dataset = Dataset.from_episodes(episodes, path=str(path), backend="zarr", **backend_options)
    return dataset, episodes


def assert_episode_equal(dataset, episode_id, episode):
    segment = dataset.episode(episode_id).read()
    for key, expected in episode["observations"].items():
        assert np.array_equal(segment[key], expected)
    # row 0 is the synthesized dummy reset row; the source's action/reward follow
    assert np.array_equal(segment["action"][1:], episode["actions"]["action"])
    assert not segment["action"][0].any()
    assert np.array_equal(segment["reward"][1:], episode["rewards"])
    assert segment["reward"][0] == 0


def test_reopen_round_trip(tmp_path):
    dataset, episodes = make_dataset(tmp_path / "ds")
    dataset.close()

    reopened = Dataset.open(str(tmp_path / "ds"))  # backend from the manifest
    assert reopened.backend.name == "zarr"
    assert reopened.num_episodes == 2
    for i, episode in enumerate(episodes):
        assert_episode_equal(reopened, i, episode)
    assert reopened.episode(0).terminated and not reopened.episode(1).terminated


def test_open_dispatches_npz_from_manifest(tmp_path):
    Dataset.from_episodes([make_episode(5)], path=str(tmp_path / "ds"))
    assert Dataset.open(str(tmp_path / "ds")).backend.name == "npz_directory"


def test_ongoing_episode_survives_flush_and_reopen(tmp_path):
    dataset, _ = make_dataset(tmp_path / "ds")
    first = make_steps(4, seed=2)
    first.pop("terminated")
    writer = dataset.new_episode()
    writer.add_steps(first)
    dataset.flush()

    reopened = Dataset.open(str(tmp_path / "ds"))
    assert reopened.episode(writer.episode_id).ongoing
    assert reopened.episode(writer.episode_id).length == 4

    second = make_steps(3, seed=3)
    second.pop("terminated")
    reopened.add_steps(writer.episode_id, second)
    reopened.end_episode(writer.episode_id, terminated=True)

    final = Dataset.open(str(tmp_path / "ds"))
    segment = final.episode(writer.episode_id).read()
    assert final.episode(writer.episode_id).length == 7
    expected = np.concatenate([first["rewards"], second["rewards"]])
    assert np.array_equal(segment["reward"], expected)
    assert final.episode(writer.episode_id).terminated


def test_crash_recovery_resets_unspilled_episode(tmp_path):
    dataset, _ = make_dataset(tmp_path / "ds")
    steps = make_steps(4, seed=2)
    steps.pop("terminated")
    writer = dataset.new_episode()
    writer.add_steps(steps)
    dataset.flush()

    pending = tmp_path / "ds" / "pending" / f"ep_{writer.episode_id:06d}.npz"
    assert pending.exists()
    os.remove(pending)  # simulate a crash that lost the spill

    reopened = Dataset.open(str(tmp_path / "ds"))
    assert reopened.episode(writer.episode_id).ongoing
    assert reopened.episode(writer.episode_id).length == 0
    for i in range(2):  # finalized episodes are untouched
        assert not reopened.episode(i).ongoing


def test_out_of_order_finalize_of_interleaved_episodes(tmp_path):
    dataset, _ = make_dataset(tmp_path / "ds")
    a = make_steps(5, seed=4)
    b = make_steps(6, seed=5)
    a.pop("terminated"), b.pop("terminated")
    writer_a = dataset.new_episode()
    writer_a.add_steps(a)
    writer_b = dataset.new_episode()
    writer_b.add_steps(b)
    dataset.end_episode(writer_b.episode_id, truncated=True)  # b before a
    dataset.end_episode(writer_a.episode_id, terminated=True)

    reopened = Dataset.open(str(tmp_path / "ds"))
    # a and b were appended as raw steps via a bare new_episode(), with no
    # synthesized reset row, so compare directly (no shift)
    for episode_id, source in ((writer_a.episode_id, a), (writer_b.episode_id, b)):
        segment = reopened.episode(episode_id).read()
        for key, expected in source["observations"].items():
            assert np.array_equal(segment[key], expected)
        assert np.array_equal(segment["action"], source["actions"]["action"])
        assert np.array_equal(segment["reward"], source["rewards"])
    assert reopened.episode(writer_b.episode_id).truncated


def test_reads_across_chunk_boundaries(tmp_path):
    # Tiny chunks force every read to span several compressed chunks.
    dataset, episodes = make_dataset(tmp_path / "ds", lengths=(50,), chunk_bytes=256)
    array = dataset.backend._group["fields/front_camera"]
    assert array.chunks[0] < 50
    for start, stop in [(0, 50), (3, 11), (17, 18), (30, 49)]:
        segment = dataset.episode(0).segment(start, stop)
        assert np.array_equal(
            segment["front_camera"], episodes[0]["observations"]["front_camera"][start:stop]
        )


def test_shard_bytes_packs_chunks(tmp_path):
    dataset, episodes = make_dataset(
        tmp_path / "ds", lengths=(40,), chunk_bytes=256, shard_bytes=4096
    )
    assert_episode_equal(dataset, 0, episodes[0])
    assert_episode_equal(Dataset.open(str(tmp_path / "ds")), 0, episodes[0])


def test_missing_optional_field_raises_keyerror(tmp_path):
    schema = DatasetSchema(
        spaces=[SpaceSpec("vec", (2,), "float32")],
        fields=[
            FieldSpec("a", "vec"),
            FieldSpec("b", "vec", optional=True),
        ],
    )
    dataset = Dataset.create(schema, path=str(tmp_path / "ds"), backend="zarr")
    values = np.ones((3, 2), dtype=np.float32)
    dataset.add_episode({"observations": {"a": values}})
    dataset.add_episode({"observations": {"a": values, "b": 2 * values}})

    for ds in (dataset, Dataset.open(str(tmp_path / "ds"))):
        assert np.array_equal(ds.backend.read_fields(["a"], Selection(0, 0, 3))["a"], values)
        with pytest.raises(KeyError):
            ds.backend.read_fields(["b"], Selection(0, 0, 3))
        assert np.array_equal(
            ds.backend.read_fields(["b"], Selection(1, 0, 3))["b"], 2 * values
        )


def test_empty_episode_finalizes(tmp_path):
    dataset, _ = make_dataset(tmp_path / "ds")
    writer = dataset.new_episode()
    dataset.end_episode(writer.episode_id, truncated=True)

    reopened = Dataset.open(str(tmp_path / "ds"))
    assert reopened.episode(writer.episode_id).length == 0
    assert reopened.episode(writer.episode_id).truncated


def test_copy_to_migrates_npz_to_zarr(tmp_path):
    episodes = [make_episode(10, seed=0), make_episode(7, seed=1, terminated=False)]
    source = Dataset.from_episodes(episodes, path=str(tmp_path / "npz"))
    ongoing = make_steps(4, seed=2)
    ongoing.pop("terminated")
    writer = source.new_episode()
    writer.add_steps(ongoing)

    copied = source.copy_to(path=str(tmp_path / "zarr"), backend="zarr")
    assert copied.backend.name == "zarr"
    assert copied.num_episodes == 3
    for i, episode in enumerate(episodes):
        assert_episode_equal(copied, i, episode)
    assert not copied.episode(0).ongoing and copied.episode(0).terminated
    assert not copied.episode(1).terminated
    assert copied.episode(writer.episode_id).ongoing  # copied still ongoing

    reopened = Dataset.open(str(tmp_path / "zarr"))
    assert reopened.num_episodes == 3
    assert_episode_equal(reopened, 0, episodes[0])
