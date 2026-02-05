# dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Callable, Union

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset

# assuming Episode is defined as in our previous step
from .episode import Episode


class EpisodeDataset:
    """
    In-memory store for Episode objects (finished or unfinished), with optional
    filesystem persistence helpers.

    Notes:
      - Keys are the Episode.episode_id values (unique per process).
      - This class does not perform sampling; a separate Sampler/Dataset wrapper
        will do that to produce (segment) minibatches.
    """

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._episodes: Dict[int, Episode] = {}
        self.storage_path = storage_path

    # ---- basic container API ----
    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[Episode]:
        # iterate in ascending order of episode_id (dict keys)
        for eid in sorted(self._episodes):
            yield self._episodes[eid]

    def ids(self) -> List[int]:
        return sorted(self._episodes.keys())

    def get(self, episode_id: int) -> Episode:
        return self._episodes[episode_id]

    def __contains__(self, episode_id: int) -> bool:
        return episode_id in self._episodes

    def add(self, ep: Episode, *, overwrite: bool = False) -> None:
        """
        Insert an Episode. By default, raises if an episode with the same ID exists.
        """
        eid = ep.episode_id
        if (not overwrite) and (eid in self._episodes):
            raise KeyError(f"Episode with id={eid} already exists in dataset.")
        self._episodes[eid] = ep

    def remove(self, episode_id: int) -> Episode:
        return self._episodes.pop(episode_id)

    # ---- convenience stats ----
    def num_complete(self) -> int:
        return sum(1 for e in self._episodes.values() if e.is_complete)

    def num_unfinished(self) -> int:
        return sum(1 for e in self._episodes.values() if not e.is_complete)

    def total_steps(self, *, only_complete: bool = True) -> int:
        if only_complete:
            return sum(e.length for e in self._episodes.values() if e.is_complete)
        return sum(e.length for e in self._episodes.values())

    def get_episode_save_path(self, episode: Episode, directory: Optional[Path] = None):
        if directory is None:
            assert self.storage_path is not None, f"Storage path not set"
            directory = self.storage_path
        return directory / "episode_{:06d}.td".format(episode.episode_id)

    # ---- persistence helpers ----
    def save(self, directory: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Save every episode to 'directory/pattern.format(episode_id)'.
        Returns a list of saved paths.

        Notes:
          - Default extension is '.td' to reflect TensorDict-based single-file saves.
          - If you prefer another extension, change `pattern` accordingly.
        """
        if directory is not None:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
        out: List[Path] = []
        for ep in self._episodes.values():
            if ep.length >= 2:
                path = self.get_episode_save_path(ep, directory)
                ep.save(path)  # Episode.save writes a single TensorDict file
                out.append(path)
        return out

    def load_dir(
            self,
            directory: Union[str, Path],
            glob: str = "*.td",
            *,
            map_location=None,
            overwrite: bool = True,
    ) -> List[int]:
        """
        Load all episode files from directory (created by Episode.save).
        Returns the list of loaded episode IDs.

        Args:
            directory: folder to scan.
            glob: filename pattern. Use '*.td' for new saves; '*.pt' only if you
                  still have legacy files and your Episode.load supports them.
            map_location: forwarded to Episode.load (e.g., 'cpu' or torch.device).
            overwrite: if True, replace any existing episode with the same ID.
        """
        directory = Path(directory)
        ids: List[int] = []
        for p in sorted(directory.glob(glob)):
            ep = Episode.load(p, map_location=map_location)
            self.add(ep, overwrite=overwrite)
            ids.append(ep.episode_id)
        return ids


class SegmentsDataset(Dataset):
    def __init__(self, episode_dataset: EpisodeDataset, segment_length: int):
        super().__init__()
        assert segment_length > 0
        self.segment_length = segment_length
        self.episode_dataset = episode_dataset

        self._total_segments = None
        self._episode_segments_cumsum = None
        self._ids = None

        self.process_dataset()

    def process_dataset(self):
        # compute the number of samples in each episode, and store a data structure
        # for look-up.
        # discard empty episodes
        episodes_num_segments = []
        ids = []
        for ep_id, ep in zip(self.episode_dataset.ids(), self.episode_dataset.__iter__()):
            l = ep.length
            if l == 0:
                continue
            num_segments = max(1, l - self.segment_length + 1)
            episodes_num_segments.append(num_segments)
            ids.append(ep_id)

        self._total_segments = np.sum(episodes_num_segments)
        self._episode_segments_cumsum = np.cumsum(episodes_num_segments)
        self._ids = ids

    def __len__(self):
        return self._total_segments if self._total_segments is not None else 0

    def __getitem__(self, index):
        if self._total_segments is None or self._episode_segments_cumsum is None:
            raise RuntimeError(f"Dataset was not pre-processed")
        if index >= self._total_segments or index < 0:
            raise ValueError(f"Index '{index}' out of bounds ({self._total_segments})")

        ep_index = np.searchsorted(self._episode_segments_cumsum, index, side='right')
        segment_index = index - self._episode_segments_cumsum[ep_index - 1] if ep_index > 0 else index
        ep_id = self._ids[ep_index]
        segment = self.episode_dataset.get(ep_id).segment(
            start=segment_index,
            end=segment_index + self.segment_length,
            should_pad=True,
        )
        return segment
