"""The storage boundary is replaceable: a third-party backend that returns
space-oriented SpaceBlocks plugs into the same logical API unchanged."""

import numpy as np

from episodata import Dataset, MemoryBackend, SpaceBlock, register_backend
from episodata.backends.base import normalize_payload
from tests.conftest import make_episode


@register_backend
class SpaceBlockBackend(MemoryBackend):
    """Memory backend that returns stacked per-space blocks instead of
    per-field arrays."""

    name = "spaceblock_memory"

    def read_fields(self, field_ids, selection):
        flat = super().read_fields(field_ids, selection)
        by_space: dict[str, list[str]] = {}
        for key in field_ids:
            by_space.setdefault(self.schema.field(key).space, []).append(key)
        return [
            SpaceBlock(
                space=space,
                keys=tuple(keys),
                data=np.stack([flat[k] for k in keys], axis=0),
            )
            for space, keys in by_space.items()
        ]


def test_space_oriented_backend_is_transparent():
    episodes = [make_episode(8, seed=0)]
    reference = Dataset.from_episodes(episodes, backend="memory")
    space_oriented = Dataset.from_episodes(episodes, backend="spaceblock_memory")

    a = reference.episode(0).segment(1, 5)
    b = space_oriented.episode(0).segment(1, 5)
    for key in a:
        assert np.array_equal(a[key], b[key])
    assert b.image.front_camera.shape == (4, 3, 8, 8)

    batch = space_oriented.segment_stream(sequence_length=3, batch_size=4, seed=0).sample()
    assert batch["front_camera"].shape == (4, 3, 3, 8, 8)


def test_normalize_payload_forms():
    arr = np.zeros((5, 2))
    assert list(normalize_payload({"a": arr})) == ["a"]
    blocks = [SpaceBlock(space="s", keys=("a", "b"), data=np.zeros((2, 5, 2)))]
    out = normalize_payload(blocks)
    assert set(out) == {"a", "b"} and out["a"].shape == (5, 2)
