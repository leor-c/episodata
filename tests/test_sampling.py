import numpy as np
import pytest

from episodata import Batch, Dataset
from tests.conftest import make_episode


def test_segment_batch_shapes(dataset):
    stream = dataset.segment_stream(
        fields=["front_camera", "state", "action"],
        sequence_length=4,
        batch_size=6,
        seed=0,
    )
    batch = stream.sample()
    assert batch.obs.front_camera.shape == (6, 4, 3, 8, 8)
    assert batch.obs["state"].shape == (6, 4, 5)
    assert batch.terminated.shape == (6, 4)


def test_context_target_segments(dataset):
    stream = dataset.segment_stream(context_length=2, target_length=3, batch_size=4, seed=0)
    batch = stream.sample()
    assert batch.obs.state.shape == (4, 5, 5)
    assert batch.context.obs.state.shape == (4, 2, 5)
    assert batch.target.obs.state.shape == (4, 3, 5)
    # target follows context contiguously
    assert np.array_equal(
        np.concatenate([batch.context.obs.state, batch.target.obs.state], axis=1),
        batch.obs.state,
    )
    # they share the boundary row: target starts where context's
    # next_observation ends
    assert np.array_equal(
        batch.target.obs.state[:, 0], batch.context.next_obs["state"][:, -1]
    )


def test_seed_determinism(dataset):
    a = dataset.segment_stream(sequence_length=3, batch_size=5, seed=42).sample()
    b = dataset.segment_stream(sequence_length=3, batch_size=5, seed=42).sample()
    assert np.array_equal(a.obs["state"], b.obs["state"])


def test_sequential_scan_covers_all_segments(dataset):
    stream = dataset.segment_stream(fields=["reward"], sequence_length=4, batch_size=3, shuffle=False)
    segments = [s for batch in stream for s in batch.reward]
    # episode lengths 10 and 7 -> (10-4+1) + (7-4+1) = 11 segments
    assert len(segments) == 11
    # rewards[i] of a segment starting at t is the source reward of step t + i
    rewards = make_episode(10, seed=0)["rewards"]
    assert np.array_equal(segments[0], rewards[0:4])
    assert np.array_equal(segments[6], rewards[6:10])


def test_terminated_flag_only_on_final_step(dataset):
    stream = dataset.segment_stream(fields=["reward"], sequence_length=4, batch_size=2, shuffle=False)
    batches = list(stream)
    flat_terminated = np.concatenate([b.terminated for b in batches], axis=0)
    # episode 0 (length 10, terminated) contributes segments 0..6; only the
    # segment ending at step 9 carries a True, on its last position.
    assert flat_terminated[:6].sum() == 0
    assert flat_terminated[6, 3] and flat_terminated[6].sum() == 1
    # episode 1 is neither terminated nor truncated
    assert flat_terminated[7:].sum() == 0


def test_transition_alignment_action_in():
    # obs t - 1 --(action/reward of step t)--> obs t, with obs 0 the reset
    t_steps = np.arange(1, 9, dtype=np.float32)
    episode = {
        "initial_observation": {"x": np.zeros(1, dtype=np.float32)},
        "observations": {"x": t_steps[:, None]},
        "actions": (10 * t_steps)[:, None],
        "rewards": 100 * t_steps,
        "terminated": True,
    }
    dataset = Dataset.from_episodes([episode])
    batch = dataset.sample_transitions(batch_size=32, seed=1)
    s_next = batch.next_obs["x"][:, 0]
    assert np.array_equal(s_next, batch.obs["x"][:, 0] + 1)
    assert np.array_equal(batch.action[:, 0], 10 * s_next)
    assert np.array_equal(batch.reward, 100 * s_next)


def test_transition_sampling(dataset):
    transitions = dataset.sample_transitions(batch_size=8, seed=0)
    assert isinstance(transitions, Batch)
    assert transitions.obs["front_camera"].shape == (8, 3, 8, 8)
    assert transitions.next_obs["state"].shape == (8, 5)
    assert transitions.action.shape == (8, 2)
    assert transitions.reward.shape == (8,)
    assert transitions.terminated.dtype == bool
    assert transitions.mask.shape == (8,)


def test_filter(dataset):
    stream = dataset.segment_stream(
        fields=["reward"],
        sequence_length=2,
        batch_size=4,
        seed=0,
        filter=lambda ep: len(ep) > 8,
    )
    scan = dataset.segment_stream(
        fields=["reward"], sequence_length=2, shuffle=False, filter=lambda ep: len(ep) > 8
    )
    # only episode 0 (length 10) passes the filter -> 9 segments
    assert sum(len(b.reward) for b in scan) == 9
    stream.sample()  # sampling under the filter works


def test_segment_too_long_raises_with_pad_disabled(dataset):
    with pytest.raises(ValueError, match="no episode"):
        dataset.segment_stream(sequence_length=100, pad=None).sample()


def test_short_episodes_sampled_with_padding(dataset):
    # segment longer than both episodes (10 and 7): each becomes one
    # zero-padded segment, so every drawn mask sums to an episode length
    stream = dataset.segment_stream(fields=["reward"], sequence_length=12, batch_size=16, seed=0)
    batch = stream.sample()
    assert batch.reward.shape == (16, 12)
    assert batch.mask.shape == (16, 12)
    assert set(batch.mask.sum(axis=1)) <= {7, 10}
    # suffix padding: zeros after the real steps
    for row in range(16):
        length = batch.mask[row].sum()
        assert batch.mask[row, :length].all()
        assert np.all(batch.reward[row, length:] == 0)


def test_prefix_padding_in_stream(dataset):
    stream = dataset.segment_stream(
        fields=["reward"], sequence_length=12, batch_size=8, seed=0, pad="prefix"
    )
    batch = stream.sample()
    for row in range(8):
        length = batch.mask[row].sum()
        assert batch.mask[row, 12 - length:].all()
        assert np.all(batch.reward[row, : 12 - length] == 0)


def test_full_segments_have_all_true_mask(dataset):
    batch = dataset.segment_stream(fields=["reward"], sequence_length=4, batch_size=5, seed=0).sample()
    assert batch.mask.all()


def test_transition_sampling_skips_short_episodes():
    # a zero-step episode has no transitions and must never be sampled
    t_steps = np.arange(1, 5, dtype=np.float32)
    long_episode = {
        "initial_observation": {"x": np.zeros(1, dtype=np.float32)},
        "observations": {"x": t_steps[:, None]},
        "actions": (10 * t_steps)[:, None],
        "rewards": 100 * t_steps,
        "terminated": True,
    }
    # just the reset observation, no steps taken
    short_episode = {
        "initial_observation": {"x": np.zeros(1, np.float32) + 50},
        "observations": {"x": np.zeros((0, 1), np.float32)},
        "actions": np.zeros((0, 1), np.float32),
        "rewards": np.zeros(0, np.float32),
        "terminated": True,
    }
    dataset = Dataset.from_episodes([long_episode, short_episode])
    batch = dataset.sample_transitions(batch_size=64, seed=0)
    assert not np.any(batch.obs["x"] == 50)


def test_online_episodes_become_sampleable(dataset):
    stream = dataset.segment_stream(fields=["reward"], sequence_length=15, batch_size=2, seed=0, pad=None)
    with pytest.raises(ValueError):
        stream.sample()
    dataset.add_episode(make_episode(20, seed=3))
    batch = stream.sample()
    assert batch.reward.shape == (2, 15)


def test_read_chunk_size_buffers_multiple_batches_with_same_shape(dataset):
    """A small read_chunk_size forces multiple internal refills across many
    sample() calls; every returned batch must still have the requested
    shape regardless of chunking."""
    stream = dataset.segment_stream(
        fields=["reward"], sequence_length=4, batch_size=3, seed=0, read_chunk_size=6
    )
    for _ in range(10):  # several refills at read_chunk_size=6, batch_size=3
        batch = stream.sample()
        assert batch.reward.shape == (3, 4)
        assert batch.mask.shape == (3, 4)


def test_read_chunk_size_default_does_not_change_correctness(dataset):
    """The default (auto) read_chunk_size batches many samples per backend
    call; each individual batch must still be internally consistent (mask
    matches real segment length) regardless of chunk grain."""
    stream = dataset.segment_stream(fields=["reward"], sequence_length=8, batch_size=4, seed=0)
    batch = stream.sample()
    for row in range(4):
        length = int(batch.mask[row].sum())
        assert batch.mask[row, :length].all()
        assert np.all(batch.reward[row, length:] == 0)


def test_custom_sampler_is_used(dataset):
    from episodata import Sampler

    class AlwaysZeroSampler:
        def sample(self, index, batch_size):
            return np.zeros(batch_size, dtype=np.int64)

    stream = dataset.segment_stream(
        fields=["reward"], sequence_length=4, batch_size=5, sampler=AlwaysZeroSampler()
    )
    batch = stream.sample()
    expected = dataset.segments(fields=["reward"], sequence_length=4)[0].reward
    for row in range(5):
        assert np.array_equal(batch.reward[row], expected)


def test_uniform_sampler_reproducible_with_seed():
    from episodata.sampler import UniformSampler

    class _FakeSegmentIndex:
        def __len__(self):
            return 100

    index = _FakeSegmentIndex()
    a = UniformSampler(seed=7).sample(index, 20)
    b = UniformSampler(seed=7).sample(index, 20)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < 100
