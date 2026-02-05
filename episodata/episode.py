# episode.py
from __future__ import annotations

from typing import Optional, List, Sequence, Union, Dict, Any
from pathlib import Path
import threading
from loguru import logger

import torch
from tensordict.tensordict import TensorDict
import tensordict as td


def _stack_steps(steps: List[TensorDict]) -> TensorDict:
    """Stack a list of scalar-batch step TensorDicts along time (batch_size=[T])."""
    if not steps:
        raise ValueError("Cannot stack empty steps list.")
    return TensorDict.stack(steps, dim=0)


class Episode:
    """
    Episode container with a consistent per-step schema and two-step backfill.

    Assumptions:
      - At least two `extend()` calls per episode:
          1) first call after reset: only {obs or next_obs}
          2) second call: full step (action, next_obs, reward, terminated, truncated)
      - After finalize(), steps are stacked into a single TensorDict with batch_size=[T]
        and the raw step list is cleared to save memory.

    Unified per-step schema (scalar batch):
      ("action", "next_obs", "reward", "terminated", "truncated", "is_first")
    """

    # ---- unique ID machinery (thread-safe) ----
    _id_lock = threading.Lock()
    _next_id: int = 0

    @classmethod
    def _allocate_id(cls) -> int:
        with cls._id_lock:
            eid = cls._next_id
            cls._next_id += 1
        return eid

    __slots__ = (
        "_device",
        "_steps",
        "_data",
        "_finalized",
        "_complete",
        "episode_id",
        "metadata",
    )

    __SEPARATOR = "."

    VERSION: str = "1.3"

    def __init__(
        self,
        episode_id: Optional[int] = None,
        *,
        device: Optional[torch.device] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.episode_id: int = episode_id if episode_id is not None else self._allocate_id()
        self.metadata: Dict[str, Any] = dict(metadata) if metadata is not None else {}

        self._device: Optional[torch.device] = device
        self._steps: List[TensorDict] = []       # list of scalar-batch steps
        self._data: Optional[TensorDict] = None  # stacked TD with batch_size=[T]
        self._finalized: bool = False
        self._complete: bool = False

    # ---- properties ----
    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def device(self) -> Optional[torch.device]:
        if self._data is not None:
            return self._data.device
        return self._device

    @property
    def length(self) -> int:
        if self._complete:
            return int(self._data.batch_size[0])  # type: ignore[union-attr]
        return len(self._steps)

    @property
    def data(self) -> TensorDict:
        if not self._complete or self._data is None:
            raise RuntimeError("Episode not yet finalized; call finalize() first.")
        return self._data

    # ---- public API ----
    def extend(self, step_like: TensorDict) -> None:
        """
        Append a step.

        - First call after reset(): accepts only {'obs' or 'next_obs'}.
                                    Stores a partial step with is_first=True.
        - Second+ calls: require full schema (action, next_obs, reward, terminated, truncated).
                         The second call triggers backfill of the first step.
        """
        if self._finalized:
            raise RuntimeError("Cannot extend a finalized episode.")

        if self.length == 0:
            self._store_partial_first(step_like)
            return

        td = self._normalize_nonfirst(step_like)
        self._steps.append(td)

        if self.length == 2:
            self._backfill_first_with_second()

        # --- NEW: auto-finalize when terminated | truncated is True on this step ---
        try:
            term = bool(td.get("terminated").item())
            trunc = bool(td.get("truncated").item())
        except Exception:
            term = trunc = False
        if term or trunc:
            self.finalize()

    def finalize(self) -> None:
        """Stack steps into a single TensorDict (batch_size=[T]) and mark complete."""
        if self._finalized:
            return
        if self.length < 2:
            raise RuntimeError("Episode expects at least two steps before finalize().")
        self._data = _stack_steps(self._steps)
        self._steps.clear()
        self._complete = True
        self._finalized = True

    def to(self, device: Union[str, torch.device]) -> "Episode":
        """Move internal tensors to the given device (pre- or post-finalize)."""
        dev = torch.device(device)
        self._device = dev
        if self._complete and self._data is not None:
            self._data = self._data.to(dev)
        else:
            for i, td in enumerate(self._steps):
                self._steps[i] = td.to(dev)
        return self

    def segment(
        self,
        start: int,
        end: int,
        *,
        drop_fields: Optional[Sequence[str]] = None,
        stack_unfinalized: bool = True,
        should_pad: bool = False,
    ) -> TensorDict:
        if start < 0 or end < 0 or end < start:
            raise ValueError(f"Invalid segment bounds: start={start}, end={end}")

        # Get a stacked view of the episode (finalized or temporary)
        if self._complete:
            td_all = self._data
            T = self.length
        else:
            T = self.length
            if not stack_unfinalized:
                raise RuntimeError("Episode not finalized; set stack_unfinalized=True to segment.")
            td_all = TensorDict.stack(self._steps, dim=0)

        span = end - start  # requested length

        if end <= T:
            seg = td_all[start:end]
            # pad_mask: all real data
            pad_mask = torch.ones(span, dtype=torch.bool, device=seg.device)
        else:
            if not should_pad:
                raise IndexError(
                    f"end={end} exceeds episode length {T}. Set should_pad=True to allow padding."
                )

            real_end = min(end, T)
            seg_real = td_all[start:real_end]

            pad_len = end - real_end
            if pad_len <= 0:
                seg = seg_real
                pad_mask = torch.ones(seg.batch_size[0], dtype=torch.bool, device=seg_real.device)
            else:
                # Build pad batch using last real step as prototype
                proto_step = td_all[T - 1]  # last real step in episode
                pad_dict = {}
                for k in seg_real.keys():
                    v = proto_step.get(k)
                    if 'obs' in k:
                        pad_v = TensorDict(
                            {
                                k_obs: torch.zeros(pad_len, *v_obs.shape, device=v_obs.device, dtype=v_obs.dtype)
                                for k_obs, v_obs in v.items()
                            },
                            batch_size=[pad_len],
                        )
                    else:
                        pad_v = torch.zeros((pad_len,) + v.shape, dtype=v.dtype, device=v.device)

                    if k == "reward":
                        pad_v = pad_v.to(torch.float32)
                    elif k == "terminated":
                        pad_v[...] = True
                    elif k == "truncated":
                        pad_v[...] = False
                    elif k == "is_first":
                        pad_v[...] = False
                    pad_dict[k] = pad_v

                seg_pad = TensorDict(pad_dict, batch_size=[pad_len])
                seg = TensorDict.cat([seg_real, seg_pad], dim=0)

                # pad_mask: ones for real part, zeros for padded tail
                pad_mask = torch.cat([
                    torch.ones(seg_real.batch_size[0], dtype=torch.bool, device=seg_real.device),
                    torch.zeros(pad_len, dtype=torch.bool, device=seg_real.device),
                ], dim=0)

        # attach pad_mask
        seg.set("pad_mask", pad_mask)

        if drop_fields:
            for k in drop_fields:
                if k in seg.keys():
                    del seg[k]

        return seg


    def save(self, path: Union[str, Path], *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save episode to a single file safely, wrapping data + metadata in an outer TensorDict.

        Structure:
            outer["data"] : time-stacked episode TensorDict  (batch_size=[T])
            outer["meta"] : scalar TensorDict with episode metadata as tensors
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # stack if needed
        if self._complete and self._data is not None:
            data_td = self._data
        else:
            if self.length < 2:
                raise RuntimeError("Cannot save: episode has fewer than two steps and is not finalized.")
            data_td = TensorDict.stack(self._steps, dim=0)

        # metadata to store as simple tensors
        meta = {
            "version": self.VERSION,
            "episode_id": torch.tensor(self.episode_id, dtype=torch.int64),
            "complete": torch.tensor(bool(self._complete)),
        }
        if self.device is not None:
            meta["device_str"] = torch.tensor(
                bytearray(str(self.device), "utf-8"), dtype=torch.uint8
            )
        # optional user metadata -> JSON bytes
        import json
        meta_str = json.dumps({**self.metadata, **(metadata or {})}, ensure_ascii=False)
        meta["metadata_json"] = torch.tensor(bytearray(meta_str, "utf-8"), dtype=torch.uint8)

        meta_td = TensorDict(meta, batch_size=[])

        # Outer TensorDict — batch_size=[] so it’s scalar
        outer = TensorDict({"data": data_td, "meta": meta_td}, batch_size=[])

        # Save safely with no pickling of custom classes
        flat = outer.flatten_keys(separator=self.__SEPARATOR)
        torch.save(flat, str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "Episode":
        path = Path(path)

        # 1) Load outer TD (no kwargs for max compatibility)
        outer = torch.load(str(path), weights_only=False).unflatten_keys(separator=cls.__SEPARATOR)

        # 2) Move to target device if requested
        if map_location is not None:
            dev = map_location if isinstance(map_location, torch.device) else torch.device(map_location)
            outer = outer.to(dev)

        data_td = outer["data"]
        meta_td = outer["meta"]

        if data_td.batch_size == torch.Size([]):
            # infer time dimension from a canonical key
            assert 'next_obs' in list(data_td.keys())
            data_td.auto_batch_size_()

        # 3) Decode metadata
        import json
        episode_id = int(meta_td["episode_id"].item())
        metadata_json = bytes(meta_td["metadata_json"].tolist()).decode("utf-8")
        meta_user = json.loads(metadata_json) if metadata_json else {}

        # 4) Build finalized Episode
        ep = cls(episode_id=episode_id, device=data_td.device, metadata=meta_user)
        ep._data = data_td
        ep._steps = []
        ep._complete = True
        ep._finalized = True
        return ep


    # ---- internals ----
    def _store_partial_first(self, step_like: TensorDict) -> None:
        """First step: require 'next_obs' or 'obs' (renamed), set is_first=True."""
        if "next_obs" in step_like.keys():
            next_obs = step_like["next_obs"].to(self._device)
        elif "obs" in step_like.keys():
            next_obs = step_like["obs"].to(self._device)
        else:
            raise KeyError("First step must contain 'next_obs' or 'obs'.")

        td0 = TensorDict({
            "next_obs": next_obs,
            "is_first": torch.ones((), dtype=torch.bool, device=self._device),
        }, batch_size=[])

        for k, v in step_like.items():
            if k not in td0:
                td0[k] = v.to(self._device)

        self._steps.append(td0)

    def _normalize_nonfirst(self, step_like: TensorDict) -> TensorDict:
        """Normalize a non-first step to the unified schema (scalar batch)."""
        required = ("action", "next_obs", "reward", "terminated", "truncated")
        missing = [k for k in required if k not in step_like.keys()]
        if missing:
            raise KeyError(f"Missing required keys for non-first step: {missing}")

        td = TensorDict({}, batch_size=[])
        td.set("action", step_like["action"].to(self._device))
        td.set("next_obs", step_like["next_obs"].to(self._device))
        td.set("reward", step_like["reward"].to(self._device).reshape(()).to(torch.float32))
        td.set("terminated", step_like["terminated"].to(self._device).reshape(()).to(torch.bool))
        td.set("truncated", step_like["truncated"].to(self._device).reshape(()).to(torch.bool))
        td.set("is_first", step_like.get("is_first", torch.zeros((), dtype=torch.bool, device=self._device))
               .to(self._device).reshape(()).to(torch.bool))
        return td

    def _backfill_first_with_second(self) -> None:
        """Fill missing fields in first step using zeros_like of second step."""
        assert self.length == 2, "backfill should be called exactly when length == 2"
        first, second = self._steps[0], self._steps[1]

        if "action" not in first.keys():
            first.set("action", torch.zeros_like(second.get("action")))
        if "reward" not in first.keys():
            first.set("reward", torch.zeros_like(second.get("reward")).to(torch.float32).reshape(()))
        if "terminated" not in first.keys():
            first.set("terminated", torch.zeros_like(second.get("terminated")).to(torch.bool).reshape(()))
        if "truncated" not in first.keys():
            first.set("truncated", torch.zeros_like(second.get("truncated")).to(torch.bool).reshape(()))
        # 'next_obs' and 'is_first' are already set
