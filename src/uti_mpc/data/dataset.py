from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.root = self.manifest_path.parent
        self.shards = manifest["shards"]
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self._cache: dict[int, tuple[np.ndarray, ...]] = {}

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative, index)
        previous = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        return shard_index, index - previous

    def _arrays(self, shard_index: int) -> tuple[np.ndarray, ...]:
        if shard_index not in self._cache:
            shard = self.shards[shard_index]
            arrays = tuple(
                np.load(self.root / shard[key], mmap_mode="r")
                for key in ("byte_tokens", "byte_mask", "length_direction", "length_mask", "labels")
            )
            self._cache[shard_index] = arrays
        return self._cache[shard_index]

    def get_label(self, index: int) -> int:
        shard_index, local = self._locate(index)
        return int(self._arrays(shard_index)[4][local])

    def labels(self) -> np.ndarray:
        return np.fromiter((self.get_label(index) for index in range(len(self))), dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local = self._locate(index)
        byte_tokens, byte_mask, lengths, length_mask, labels = self._arrays(shard_index)
        shard = self.shards[shard_index]
        flow_id = f"{shard['id']}:{local}"
        return {
            "byte_tokens": torch.as_tensor(np.array(byte_tokens[local]), dtype=torch.long),
            "byte_mask": torch.as_tensor(np.array(byte_mask[local]), dtype=torch.bool),
            "length_direction": torch.as_tensor(np.array(lengths[local]), dtype=torch.float32),
            "length_mask": torch.as_tensor(np.array(length_mask[local]), dtype=torch.bool),
            "label": torch.tensor(int(labels[local]), dtype=torch.long),
            "flow_id": flow_id,
        }

