"""The storage boundary is replaceable: a third-party backend registered via
``register_backend`` plugs into the same logical API unchanged."""

import numpy as np

from episodata import Dataset, MemoryBackend, register_backend
from episodata.backends.base import Selection
from tests.conftest import make_episode, make_steps


@register_backend
class ThirdPartyBackend(MemoryBackend):
    """Memory backend under its own name, standing in for an external plugin."""

    name = "third_party_memory"


def test_registered_backend_is_transparent():
    episodes = [make_episode(8, seed=0)]
    reference = Dataset.from_episodes(episodes, backend="memory")
    plugged = Dataset.from_episodes(episodes, backend="third_party_memory")

    a = reference.episode(0).segment(1, 5)
    b = plugged.episode(0).segment(1, 5)
    assert set(a.obs) == set(b.obs)
    for key in a.obs:
        assert np.array_equal(a.obs[key], b.obs[key])
    assert np.array_equal(a.action, b.action)
    assert np.array_equal(a.reward, b.reward)

    batch = plugged.segment_stream(sequence_length=3, batch_size=4, seed=0).sample()
    assert batch.obs.front_camera.shape == (4, 3, 3, 8, 8)


def test_append_steps_batch_default_matches_looped():
    # the default implementation is inherited by any backend, including
    # third-party subclasses that never heard of it
    for backend in ("memory", "third_party_memory"):
        dataset = Dataset.from_episodes([make_episode(3, seed=0)], backend=backend)
        ids = [dataset.new_episode().episode_id for _ in range(2)]
        revision = dataset.backend.revision

        source = make_steps(2, seed=1)
        rows = {
            "front_camera": source["observations"]["front_camera"],
            "wrist_camera": source["observations"]["wrist_camera"],
            "state": source["observations"]["state"],
            "action": source["actions"]["action"],
            "reward": source["rewards"],
        }
        dataset.backend.append_steps_batch(ids, rows)

        assert dataset.backend.revision > revision
        for i, episode_id in enumerate(ids):
            assert dataset.backend.episode_length(episode_id) == 1
            # a backend-level check: read the raw row across the boundary
            read = dataset.backend.read_fields(
                ["state", "reward"], Selection(episode_id, 0, 1)
            )
            assert np.array_equal(read["state"][0], rows["state"][i])
            assert read["reward"][0] == rows["reward"][i]
