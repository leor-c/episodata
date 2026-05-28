# episode.py
from __future__ import annotations

from typing import Optional, List, Sequence, Union, Dict, Any
from pathlib import Path
import threading

import torch
from tensordict.tensordict import TensorDict


def _stack_steps(steps: List[TensorDict]) -> TensorDict:
    """Stack a list of scalar-batch step TensorDicts along time (batch_size=[T])."""
    if not steps:
        raise ValueError("Cannot stack empty steps list.")
    return TensorDict.stack(steps, dim=0)


class Episode:
    """
    Episode container with a consistent per-step schema and two-step backfill.

    Internal storage after finalize(): a flat dict[str, Tensor] with dotted keys
    (e.g. 'next_obs.image.frame'). This lets SegmentsDataset.__getitem__ do
    plain tensor slicing with no TensorDict overhead in the hot path.

    Public API still returns TensorDicts; `data` and `segment()` reconstruct them
    from the flat dict on demand.

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
        "_flat",        # dict[str, Tensor] with dotted keys; set after finalize()
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
        self._steps: List[TensorDict] = []
        self._flat: Optional[Dict[str, torch.Tensor]] = None
        self._finalized: bool = False
        self._complete: bool = False

    # ---- properties ----
    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def length(self) -> int:
        if self._complete:
            return next(iter(self._flat.values())).shape[0]
        return len(self._steps)

    @property
    def device(self) -> Optional[torch.device]:
        if self._complete and self._flat:
            return next(iter(self._flat.values())).device
        return self._device

    @property
    def data(self) -> TensorDict:
        """Full episode as TensorDict (backward-compat). Reconstructed from flat storage."""
        if not self._complete:
            raise RuntimeError("Episode not yet finalized; call finalize() first.")
        return TensorDict(self._flat, batch_size=[self.length], device=self.device).unflatten_keys(self.__SEPARATOR)

    @property
    def all_data(self) -> Dict[str, torch.Tensor]:
        """
        Flat dict[str, Tensor] with dotted keys, regardless of finalization state.
        For finalized episodes this is a direct reference (no copy).
        For unfinalized episodes the steps are stacked on the fly — avoid calling
        in the training hot path.
        """
        if self._complete:
            return self._flat
        stacked = TensorDict.stack(self._steps, dim=0)
        return dict(stacked.flatten_keys(self.__SEPARATOR).items())

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

        try:
            term = bool(td.get("terminated").item())
            trunc = bool(td.get("truncated").item())
        except (RuntimeError, ValueError):
            term = trunc = False
        if term or trunc:
            self.finalize()

    def finalize(self) -> None:
        """Stack steps into flat dict[str, Tensor] and mark complete."""
        if self._finalized:
            return
        if self.length < 2:
            raise RuntimeError("Episode expects at least two steps before finalize().")
        stacked = _stack_steps(self._steps)
        self._flat = dict(stacked.flatten_keys(self.__SEPARATOR).items())
        self._steps.clear()
        self._complete = True
        self._finalized = True

    def to(self, device: Union[str, torch.device]) -> "Episode":
        """Move internal tensors to the given device (pre- or post-finalize)."""
        dev = torch.device(device)
        self._device = dev
        if self._complete:
            self._flat = {k: v.to(dev) for k, v in self._flat.items()}
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

        if not self._complete and not stack_unfinalized:
            raise RuntimeError("Episode not finalized; set stack_unfinalized=True to segment.")

        flat = self.all_data
        T = self.length
        span = end - start
        real_end = min(end, T)
        pad_len = end - real_end

        if pad_len > 0 and not should_pad:
            raise IndexError(
                f"end={end} exceeds episode length {T}. Set should_pad=True to allow padding."
            )

        def _slice(k: str, v: torch.Tensor) -> torch.Tensor:
            real = v[start:real_end]
            if pad_len == 0:
                return real
            pad = real.new_zeros((pad_len,) + real.shape[1:])
            leaf = k.rsplit(self.__SEPARATOR, 1)[-1]
            if leaf == "terminated":
                pad = pad.bool().fill_(True)
            return torch.cat([real, pad], dim=0)

        sliced: Dict[str, torch.Tensor] = {k: _slice(k, v) for k, v in flat.items()}

        first_val = sliced[next(iter(sliced))]
        pad_mask = torch.zeros(span, dtype=torch.bool, device=first_val.device)
        pad_mask[:real_end - start] = True
        sliced["pad_mask"] = pad_mask

        if drop_fields:
            sep = self.__SEPARATOR
            for field in drop_fields:
                for k in [k for k in list(sliced) if k == field or k.startswith(field + sep)]:
                    del sliced[k]

        dev = sliced[next(iter(sliced))].device
        return TensorDict(sliced, batch_size=[span], device=dev).unflatten_keys(self.__SEPARATOR)

    def save(self, path: Union[str, Path], *, metadata: Optional[Dict[str, Any]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._complete:
            data_td = TensorDict(self._flat, batch_size=[self.length], device=self.device).unflatten_keys(self.__SEPARATOR)
        else:
            if self.length < 2:
                raise RuntimeError("Cannot save: episode has fewer than two steps and is not finalized.")
            data_td = TensorDict.stack(self._steps, dim=0)

        meta = {
            "version": torch.tensor(bytearray(self.VERSION, "utf-8"), dtype=torch.uint8),
            "episode_id": torch.tensor(self.episode_id, dtype=torch.int64),
            "complete": torch.tensor(bool(self._complete)),
        }
        if self.device is not None:
            meta["device_str"] = torch.tensor(
                bytearray(str(self.device), "utf-8"), dtype=torch.uint8
            )
        import json
        meta_str = json.dumps({**self.metadata, **(metadata or {})}, ensure_ascii=False)
        meta["metadata_json"] = torch.tensor(bytearray(meta_str, "utf-8"), dtype=torch.uint8)

        meta_td = TensorDict(meta, batch_size=[])
        outer = TensorDict({"data": data_td, "meta": meta_td}, batch_size=[])
        torch.save(outer.flatten_keys(separator=self.__SEPARATOR), str(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        *,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "Episode":
        path = Path(path)

        outer = torch.load(str(path), weights_only=False).unflatten_keys(separator=cls.__SEPARATOR)

        if map_location is not None:
            dev = map_location if isinstance(map_location, torch.device) else torch.device(map_location)
            outer = outer.to(dev)

        data_td = outer["data"]
        meta_td = outer["meta"]

        if data_td.batch_size == torch.Size([]):
            assert 'next_obs' in list(data_td.keys())
            data_td.auto_batch_size_()

        import json
        episode_id = int(meta_td["episode_id"].item())
        metadata_json = bytes(meta_td["metadata_json"].tolist()).decode("utf-8")
        meta_user = json.loads(metadata_json) if metadata_json else {}

        ep = cls(episode_id=episode_id, metadata=meta_user)
        ep._flat = dict(data_td.flatten_keys(cls.__SEPARATOR).items())
        ep._device = next(iter(ep._flat.values())).device
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
            if k not in td0 and k != "obs":
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
