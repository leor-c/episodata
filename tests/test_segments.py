import numpy as np
import pytest

from tests.conftest import make_episode


def test_len_and_getitem_shapes(dataset):
    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    # episode lengths 10 and 7 -> (10-4+1) + (7-4+1) = 11 segments
    assert len(segments) == 11
    segment = segments[0]
    assert segment["front_camera"].shape == (4, 3, 8, 8)
    assert segment.image.front_camera.shape == (4, 3, 8, 8)
    assert segment["state"].shape == (4, 5)
    assert segment.terminated.shape == (4,)
    assert segment.truncated.shape == (4,)


def test_getitem_out_of_range_raises(dataset):
    segments = dataset.segments(fields=["reward"], sequence_length=4)
    with pytest.raises(IndexError):
        segments[len(segments)]


def test_matches_sequential_loader_scan(dataset):
    """SegmentDataset and Loader share SegmentIndex/read_segment, so a
    sequential (unshuffled) Loader scan and direct segment[i] access must
    agree segment-for-segment."""
    loader = dataset.loader(fields=["reward"], sequence_length=4, batch_size=3, shuffle=False)
    scanned = [s for batch in loader for s in batch["reward"]]
    segments = dataset.segments(fields=["reward"], sequence_length=4)
    assert len(segments) == len(scanned)
    for i, expected in enumerate(scanned):
        assert np.array_equal(segments[i]["reward"], expected)


def test_terminated_flag_only_on_final_step(dataset):
    segments = dataset.segments(fields=["reward"], sequence_length=4)
    # episode 0 (length 10, terminated) contributes segments 0..6; only the
    # segment ending at step 9 carries a True, on its last position.
    for i in range(6):
        assert segments[i].terminated.sum() == 0
    assert segments[6].terminated[3] and segments[6].terminated.sum() == 1
    # episode 1 is neither terminated nor truncated
    for i in range(7, len(segments)):
        assert segments[i].terminated.sum() == 0
        assert segments[i].truncated.sum() == 0


def test_collate_matches_loader_batch_shape(dataset):
    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    items = [segments[i] for i in (6, 0, 7)]
    batch = segments.collate(items)
    assert batch["front_camera"].shape == (3, 4, 3, 8, 8)
    assert batch["state"].shape == (3, 4, 5)
    assert batch.terminated.shape == (3, 4)
    assert batch.terminated[0, 3]  # item 6 carries the terminal flag
    assert batch.terminated[1:].sum() == 0
    for row, item in enumerate(items):
        assert np.array_equal(batch["state"][row], item["state"])


def test_collate_context_target(dataset):
    segments = dataset.segments(context_length=2, target_length=3)
    items = [segments[i] for i in range(4)]
    batch = segments.collate(items)
    assert batch["state"].shape == (4, 5, 5)
    assert batch.context["state"].shape == (4, 2, 5)
    assert batch.target["state"].shape == (4, 3, 5)
    assert np.array_equal(
        np.concatenate([batch.context["state"], batch.target["state"]], axis=1),
        batch["state"],
    )


def test_batch_role_views(dataset):
    segments = dataset.segments(sequence_length=4)
    batch = segments.collate([segments[0], segments[1]])
    assert set(batch.actions) == {"action"}
    assert batch.actions["action"].shape == (2, 4, 2)
    assert np.array_equal(batch.actions.mask, batch.mask)


def test_filter(dataset):
    segments = dataset.segments(
        fields=["reward"], sequence_length=2, filter=lambda ep: len(ep) > 8
    )
    # only episode 0 (length 10) passes the filter -> 9 segments
    assert len(segments) == 9


def test_empty_index_when_segment_too_long_and_pad_disabled(dataset):
    segments = dataset.segments(sequence_length=100, pad=None)
    assert len(segments) == 0
    with pytest.raises(IndexError):
        segments[0]


def test_short_episode_yields_one_suffix_padded_segment(dataset):
    # episode lengths 10 and 7 -> ep0 gives 3 full segments, ep1 one padded
    segments = dataset.segments(fields=["reward"], sequence_length=8)
    assert len(segments) == 4
    padded = segments[3]
    source = make_episode(7, seed=1, terminated=False)["rewards"]
    assert np.array_equal(padded["reward"][:7], source)
    assert np.array_equal(padded["reward"][7:], np.zeros(1, dtype=np.float32))
    assert np.array_equal(padded.mask, [True] * 7 + [False])
    # full segments carry an all-True mask
    assert segments[0].mask.all()


def test_short_episode_prefix_padding(dataset):
    segments = dataset.segments(fields=["reward"], sequence_length=8, pad="prefix")
    padded = segments[3]
    source = make_episode(7, seed=1, terminated=False)["rewards"]
    assert np.array_equal(padded["reward"][1:], source)
    assert padded["reward"][0] == 0
    assert np.array_equal(padded.mask, [False] + [True] * 7)


def test_padded_segment_terminal_flag_position(dataset):
    # segment longer than both episodes -> each gives one padded segment;
    # episode 0 (length 10) is terminated, so the flag sits on its last
    # real step: offset 9 with suffix padding, segment-1 with prefix.
    suffix = dataset.segments(fields=["reward"], sequence_length=12)
    assert len(suffix) == 2
    assert suffix[0].terminated[9] and suffix[0].terminated.sum() == 1
    assert suffix[1].terminated.sum() == 0  # episode 1 not terminated
    prefix = dataset.segments(fields=["reward"], sequence_length=12, pad="prefix")
    assert prefix[0].terminated[11] and prefix[0].terminated.sum() == 1


def test_collate_carries_mask(dataset):
    segments = dataset.segments(fields=["reward"], sequence_length=8)
    batch = segments.collate([segments[0], segments[3]])
    assert batch.mask.shape == (2, 8)
    assert batch.mask[0].all()
    assert np.array_equal(batch.mask[1], [True] * 7 + [False])


def test_invalid_pad_value_raises(dataset):
    with pytest.raises(ValueError, match="pad"):
        dataset.segments(sequence_length=4, pad="middle")


def test_torch_dataloader_integration(dataset):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    loader = DataLoader(
        segments, batch_size=6, shuffle=True, num_workers=0, collate_fn=segments.collate
    )
    batch = next(iter(loader))
    assert batch["front_camera"].shape == (6, 4, 3, 8, 8)
    assert batch["state"].shape == (6, 4, 5)


def test_torch_dataloader_multiprocess(backend_name, dataset_path):
    """Real multi-worker read parallelism only matters for a disk-backed
    backend (npz_directory does synchronous decompression per read)."""
    if backend_name != "npz_directory":
        pytest.skip("multiprocess benefit is specific to disk-backed backends")
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from episodata import Dataset

    episodes = [make_episode(10, seed=0), make_episode(7, seed=1, terminated=False)]
    ds = Dataset.from_episodes(episodes, path=dataset_path, backend=backend_name)
    segments = ds.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    loader = DataLoader(
        segments, batch_size=4, shuffle=True, num_workers=2, collate_fn=segments.collate
    )
    batches = [batch for _, batch in zip(range(3), loader)]
    assert len(batches) == 3
    assert batches[0]["front_camera"].shape == (4, 4, 3, 8, 8)
