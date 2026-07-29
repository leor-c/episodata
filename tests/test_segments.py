import numpy as np
import pytest

from tests.conftest import make_episode


def test_len_and_getitem_shapes(dataset):
    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    # episode lengths 10 and 7 -> (10-4+1) + (7-4+1) = 11 segments
    assert len(segments) == 11
    segment = segments[0]
    assert segment.obs.front_camera.shape == (4, 3, 8, 8)
    assert segment.obs["state"].shape == (4, 5)
    assert segment.terminated.shape == (4,)
    assert segment.truncated.shape == (4,)


def test_getitem_out_of_range_raises(dataset):
    segments = dataset.segments(fields=["reward"], sequence_length=4)
    with pytest.raises(IndexError):
        segments[len(segments)]


def test_matches_sequential_stream_scan(dataset):
    """SegmentStream draws its segments from SegmentDataset, so a sequential
    (unshuffled) SegmentStream scan and direct segment[i] access must agree
    segment-for-segment."""
    stream = dataset.segment_stream(fields=["reward"], sequence_length=4, batch_size=3, shuffle=False)
    scanned = [s for batch in stream for s in batch.reward]
    segments = dataset.segments(fields=["reward"], sequence_length=4)
    assert len(segments) == len(scanned)
    for i, expected in enumerate(scanned):
        assert np.array_equal(segments[i].reward, expected)


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


def test_collate_matches_stream_batch_shape(dataset):
    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    items = [segments[i] for i in (6, 0, 7)]
    batch = segments.collate(items)
    assert batch.obs.front_camera.shape == (3, 4, 3, 8, 8)
    assert batch.obs["state"].shape == (3, 4, 5)
    assert batch.terminated.shape == (3, 4)
    assert batch.terminated[0, 3]  # item 6 carries the terminal flag
    assert batch.terminated[1:].sum() == 0
    for row, item in enumerate(items):
        assert np.array_equal(batch.obs["state"][row], item.obs["state"])


def test_collate_context_target(dataset):
    segments = dataset.segments(context_length=2, target_length=3)
    items = [segments[i] for i in range(4)]
    batch = segments.collate(items)
    assert batch.obs.state.shape == (4, 5, 5)
    assert batch.context.obs.state.shape == (4, 2, 5)
    assert batch.target.obs.state.shape == (4, 3, 5)
    assert np.array_equal(
        np.concatenate([batch.context.obs.state, batch.target.obs.state], axis=1),
        batch.obs.state,
    )


def test_batch_role_views(dataset):
    segments = dataset.segments(sequence_length=4)
    batch = segments.collate([segments[0], segments[1]])
    assert set(batch.observations) == {"front_camera", "wrist_camera", "state"}
    assert batch.action.shape == (2, 4, 2)
    assert batch.actions is batch.action
    assert batch.mask.shape == (2, 4)


def test_all_observations_is_the_shared_buffer(dataset):
    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    seg = segments[0]
    rows = seg.all_observations
    assert rows["state"].shape == (5, 5)
    assert np.array_equal(rows["state"][:-1], seg.obs["state"])
    assert np.array_equal(rows["state"][1:], seg.next_obs["state"])
    assert np.shares_memory(rows["state"], seg.obs["state"])
    assert np.shares_memory(rows["state"], seg.next_obs["state"])
    assert seg.all_obs is seg.all_observations
    # action fields never leak into the observation rows
    assert set(rows) == {"front_camera", "state"}


def test_batch_all_observations_shape(dataset):
    segments = dataset.segments(fields=["state", "action"], sequence_length=4)
    batch = segments.collate([segments[0], segments[1]])
    assert batch.all_observations["state"].shape == (2, 5, 5)
    assert np.array_equal(batch.all_observations["state"][:, :-1], batch.obs["state"])


class FakeTensor:
    """Minimal non-numpy array: basic slicing and shape only."""

    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, index):
        return FakeTensor(self.arr[index])

    @property
    def shape(self):
        return self.arr.shape


def test_map_rederives_views_from_converted_buffers(dataset):
    segments = dataset.segments(fields=["state", "action"], sequence_length=4)
    seg = segments[0]
    converted = []

    def fn(a):
        copy = np.array(a)
        converted.append(copy)
        return copy

    mapped = seg.map(fn)
    # fn ran once per field buffer plus the three flags — not once per view
    assert len(converted) == 2 + 3
    assert np.array_equal(mapped.obs["state"], seg.obs["state"])
    assert np.array_equal(mapped.next_obs["state"], seg.next_obs["state"])
    assert np.array_equal(mapped.action, seg.action)
    assert np.array_equal(mapped.mask, seg.mask)
    # obs and next_obs of the result still share one converted buffer
    assert np.shares_memory(mapped.obs["state"], mapped.next_obs["state"])
    assert not np.shares_memory(mapped.obs["state"], seg.obs["state"])


def test_map_supports_non_numpy_arrays(dataset):
    segments = dataset.segments(fields=["state", "action"], sequence_length=4)
    seg = segments[0]
    mapped = seg.map(FakeTensor)
    assert np.array_equal(mapped.obs["state"].arr, seg.obs["state"])
    assert np.array_equal(mapped.next_obs["state"].arr, seg.next_obs["state"])
    assert np.array_equal(mapped.action.arr, seg.action)


def test_map_batch_preserves_context_target(dataset):
    segments = dataset.segments(fields=["state", "action"], context_length=2, target_length=3)
    batch = segments.collate([segments[0], segments[1]])
    mapped = batch.map(np.array)
    assert type(mapped) is type(batch)
    assert mapped.context.obs["state"].shape == (2, 2, 5)
    assert mapped.target.obs["state"].shape == (2, 3, 5)
    assert np.array_equal(mapped.terminated, batch.terminated)


def test_map_to_torch_shares_storage(dataset):
    torch = pytest.importorskip("torch")

    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    batch = segments.collate([segments[0], segments[1]])
    mapped = batch.map(lambda a: torch.as_tensor(np.ascontiguousarray(a)))
    obs = mapped.obs["front_camera"]
    next_obs = mapped.next_obs["front_camera"]
    assert isinstance(obs, torch.Tensor)
    assert obs.untyped_storage().data_ptr() == next_obs.untyped_storage().data_ptr()
    assert np.array_equal(obs.numpy(), batch.obs["front_camera"])


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
    rewards = make_episode(7, seed=1, terminated=False)["rewards"]
    assert np.array_equal(padded.reward[:7], rewards)
    assert np.array_equal(padded.reward[7:], np.zeros(1, dtype=np.float32))
    assert np.array_equal(padded.mask, [True] * 7 + [False])
    # full segments carry an all-True mask
    assert segments[0].mask.all()


def test_short_episode_prefix_padding(dataset):
    segments = dataset.segments(fields=["reward"], sequence_length=8, pad="prefix")
    padded = segments[3]
    rewards = make_episode(7, seed=1, terminated=False)["rewards"]
    assert np.array_equal(padded.reward[1:], rewards)
    assert padded.reward[0] == 0  # the padded transition is masked out
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


def test_refresh_reveals_appended_episodes(dataset):
    segments = dataset.segments(fields=["reward"], sequence_length=4)
    assert len(segments) == 11
    dataset.add_episode(make_episode(6, seed=2))
    # the snapshot is stable until an explicit refresh
    assert len(segments) == 11
    segments.refresh()
    assert len(segments) == 11 + 3
    rewards = make_episode(6, seed=2)["rewards"]
    assert np.array_equal(segments[11].reward, rewards[0:4])


def test_refresh_is_noop_without_writes(dataset):
    segments = dataset.segments(fields=["reward"], sequence_length=4)
    index = segments._index
    segments.refresh()
    assert segments._index is index  # not rebuilt: backend saw no writes
    dataset.add_episode(make_episode(6, seed=2))
    segments.refresh()
    assert segments._index is not index


def test_refresh_respects_filter(dataset):
    segments = dataset.segments(
        fields=["reward"], sequence_length=2, filter=lambda ep: len(ep) > 8
    )
    assert len(segments) == 9
    dataset.add_episode(make_episode(4, seed=2))  # too short for the filter
    dataset.add_episode(make_episode(10, seed=3))
    segments.refresh()
    assert len(segments) == 9 + 9


def test_torch_dataloader_integration(dataset):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    loader = DataLoader(
        segments, batch_size=6, shuffle=True, num_workers=0, collate_fn=segments.collate
    )
    batch = next(iter(loader))
    assert batch.obs.front_camera.shape == (6, 4, 3, 8, 8)
    assert batch.obs["state"].shape == (6, 4, 5)


def test_fetch_matches_looped_getitem(dataset):
    """SegmentDataset.fetch is a vectorized equivalent of collate([self[i]
    for i in indices]); must agree exactly, including duplicate indices,
    out-of-order indices, and multiple episodes in one call."""
    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    idx = np.array([6, 0, 0, 10, 3, 6])  # duplicates + out of order + spans both episodes
    via_fetch = segments.fetch(idx)
    via_loop = segments.collate([segments[int(i)] for i in idx])
    for key in ("front_camera", "state"):
        assert np.array_equal(via_fetch.obs[key], via_loop.obs[key])
    assert np.array_equal(via_fetch.action, via_loop.action)
    assert np.array_equal(via_fetch.terminated, via_loop.terminated)
    assert np.array_equal(via_fetch.truncated, via_loop.truncated)
    assert np.array_equal(via_fetch.mask, via_loop.mask)


def test_fetch_matches_looped_getitem_with_padding(dataset):
    """Boundary/short-episode indices needing padding, both directions."""
    for pad in ("suffix", "prefix"):
        segments = dataset.segments(fields=["reward"], sequence_length=8, pad=pad)
        idx = np.array([3, 1, 3, 0])  # index 3 is the padded short-episode segment
        via_fetch = segments.fetch(idx)
        via_loop = segments.collate([segments[int(i)] for i in idx])
        assert np.array_equal(via_fetch.reward, via_loop.reward)
        assert np.array_equal(via_fetch.mask, via_loop.mask)
        assert np.array_equal(via_fetch.terminated, via_loop.terminated)


def test_fetch_random_with_replacement_matches_looped_getitem(dataset):
    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    idx = np.random.default_rng(0).integers(len(segments), size=50)
    via_fetch = segments.fetch(idx)
    via_loop = segments.collate([segments[int(i)] for i in idx])
    assert np.array_equal(via_fetch.obs["state"], via_loop.obs["state"])
    assert np.array_equal(via_fetch.mask, via_loop.mask)


def test_getitems_matches_looped_getitem(dataset):
    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    idx = [6, 0, 0, 10, 3]
    batched = segments.__getitems__(idx)
    looped = [segments[i] for i in idx]
    assert len(batched) == len(looped)
    for a, b in zip(batched, looped):
        assert np.array_equal(a.obs["state"], b.obs["state"])
        assert np.array_equal(a.mask, b.mask)
        assert np.array_equal(a.terminated, b.terminated)


def test_getitems_dataloader_output_unchanged(dataset):
    """DataLoader picks up __getitems__ automatically (torch's batched-fetch
    hook) — output must be identical to the pre-existing per-item path."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    segments = dataset.segments(fields=["front_camera", "state", "action"], sequence_length=4)
    loader = DataLoader(
        segments, batch_size=6, shuffle=False, num_workers=0, collate_fn=segments.collate
    )
    batch = next(iter(loader))
    expected = segments.collate([segments[i] for i in range(6)])
    assert np.array_equal(batch.obs["front_camera"], expected.obs["front_camera"])
    assert np.array_equal(batch.obs["state"], expected.obs["state"])
    assert np.array_equal(batch.mask, expected.mask)


def test_resolve_many_matches_looped_resolve(dataset):
    segments = dataset.segments(fields=["reward"], sequence_length=4)
    idx = np.array([6, 0, 10, 3, 6, 1])
    selections, terminal_offsets = segments._index.resolve_many(idx)
    for i, flat_idx in enumerate(idx):
        expected_selection, expected_terminal_offset = segments._index.resolve(int(flat_idx))
        assert selections[i] == expected_selection
        assert terminal_offsets[i] == expected_terminal_offset


def test_read_fields_batch_matches_looped_single_selection(dataset):
    from episodata.backends.base import Selection

    backend = dataset.backend
    selections = [Selection(0, 0, 4), Selection(1, 0, 4), Selection(0, 0, 4), Selection(1, 3, 6)]
    batched = backend.read_fields(["reward"], selections)
    looped = [backend.read_fields(["reward"], [s])[0] for s in selections]
    assert len(batched) == len(looped)
    for a, b in zip(batched, looped):
        assert np.array_equal(a["reward"], b["reward"])


def test_read_fields_mixed_ongoing_and_finalized(backend_name, dataset_path):
    """A batch spanning both an already-finalized episode and a still-open
    (buffered) one must recombine results in original order."""
    from episodata import Dataset
    from episodata.backends.base import Selection

    episodes = [make_episode(10, seed=0)]
    ds = Dataset.from_episodes(episodes, path=dataset_path, backend=backend_name)
    ongoing = make_episode(5, seed=7)
    writer = ds.new_episode(ongoing["initial_observation"])
    steps = {
        "observations": ongoing["observations"],
        "actions": ongoing["actions"],
        "rewards": ongoing["rewards"],
    }
    writer.add_steps(steps)
    assert ds.episode(1).ongoing

    selections = [Selection(1, 0, 4), Selection(0, 0, 4), Selection(1, 2, 5), Selection(0, 6, 9)]
    results = ds.backend.read_fields(["reward"], selections)
    for position, selection in enumerate(selections):
        expected = ds.backend.read_fields(["reward"], [selection])[0]
        assert np.array_equal(results[position]["reward"], expected["reward"])


def test_read_fields_keyerror_parity(backend_name, dataset_path):
    from episodata import Dataset, DatasetSchema, FieldSpec
    from episodata.backends.base import Selection

    schema = DatasetSchema(
        fields=[
            FieldSpec("a", (2,), "float32"),
            FieldSpec("b", (2,), "float32", optional=True),
        ]
    )
    ds = Dataset.create(schema, path=dataset_path, backend=backend_name)
    values = np.ones((3, 2), dtype=np.float32)
    initial = np.full(2, 7, dtype=np.float32)
    ds.add_episode({"initial_observation": {"a": initial}, "observations": {"a": values}})

    with pytest.raises(KeyError):
        ds.backend.read_fields(["b"], [Selection(0, 1, 4)])


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
    assert batches[0].obs.front_camera.shape == (4, 4, 3, 8, 8)
