"""End-to-end example: build, refine, persist, reopen, sample."""

import numpy as np

from episodata import Dataset


def make_episode(length: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "observations": {
            "front_camera": rng.integers(0, 256, size=(length, 3, 64, 64), dtype=np.uint8),
            "wrist_camera": rng.integers(0, 256, size=(length, 3, 64, 64), dtype=np.uint8),
            "state": rng.standard_normal((length, 7)).astype(np.float32),
        },
        "actions": rng.standard_normal((length, 4)).astype(np.float32),
        "rewards": rng.standard_normal(length).astype(np.float32),
        "terminated": True,
    }


if __name__ == "__main__":
    # Automatic schema inference from example data, persisted via the
    # npz-directory backend (drop `path` to stay in memory).
    dataset = Dataset.from_episodes(
        [make_episode(40, seed=0), make_episode(25, seed=1)],
        path="example_dataset",
    )
    print(dataset)
    print(dataset.schema)

    # Hybrid mode: refine an inferred space key; the persisted schema is
    # authoritative from now on.
    dataset.rename_space("vector", "proprio")

    # Episode and observation access.
    episode = dataset.episode(0)
    obs = episode.segment(0, 8, fields=["front_camera", "state", "action"])
    print("segment:", obs.image.front_camera.shape, obs.proprio.state.shape)

    # Online append, mirroring the Gym loop: the episode begins at reset,
    # then one row per env.step (the action sent plus what it produced).
    writer = dataset.new_episode(
        {
            "front_camera": np.zeros((3, 64, 64), dtype=np.uint8),
            "wrist_camera": np.zeros((3, 64, 64), dtype=np.uint8),
            "state": np.zeros(7, dtype=np.float32),
        }
    )
    print("ongoing:", dataset.episode(writer.episode_id).ongoing)
    for t in range(10):
        writer.add_step(
            {
                "observations": {
                    "front_camera": np.zeros((3, 64, 64), dtype=np.uint8),
                    "wrist_camera": np.zeros((3, 64, 64), dtype=np.uint8),
                    "state": np.zeros(7, dtype=np.float32),
                },
                "actions": np.zeros(4, dtype=np.float32),
                "rewards": 1.0,
                # separate Gymnasium-style signals; True finalizes the episode
                "terminated": t == 9,
                "truncated": False,
            }
        )
    print("terminated:", dataset.episode(writer.episode_id).terminated)

    # Reopen: same logical API, schema not re-inferred.
    dataset = Dataset.open("example_dataset")

    # Context/target segments for world-model training.
    stream = dataset.segment_stream(
        fields=["front_camera", "state", "action"],
        context_length=4,
        target_length=12,
        batch_size=16,
        seed=0,
    )
    batch = stream.sample()
    print("batch:", batch.image.front_camera.shape)
    print("context/target:", batch.context["state"].shape, batch.target["state"].shape)

    # Transition sampling for control.
    transitions = dataset.sample_transitions(batch_size=32, seed=0)
    print("transitions:", transitions.observations["front_camera"].shape,
          transitions.rewards.shape, transitions.terminated.sum(), "terminal")
