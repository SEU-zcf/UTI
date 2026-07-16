from __future__ import annotations

import bisect
import hashlib
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
        self.manifest = manifest
        self.representation = str(manifest.get("representation", "legacy"))
        self.is_v3 = self.representation == "uti_mpc_v3"
        self.fingerprint = hashlib.sha256(
            self.manifest_path.read_bytes()
        ).hexdigest()
        self.root = self.manifest_path.parent
        self.shards = manifest["shards"]
        self.cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard["count"])
            self.cumulative.append(total)
        self._cache: dict[int, dict[str, np.ndarray]] = {}

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

    def _array_names(self) -> tuple[str, ...]:
        if self.is_v3:
            return (
                "payload_tokens",
                "payload_mask",
                "packet_features",
                "packet_mask",
                "burst_features",
                "burst_mask",
                "labels",
            )
        return ("byte_tokens", "byte_mask", "length_direction", "length_mask", "labels")

    def _arrays(self, shard_index: int) -> dict[str, np.ndarray]:
        if shard_index not in self._cache:
            shard = self.shards[shard_index]
            arrays = {
                key: np.load(self.root / shard[key], mmap_mode="r")
                for key in self._array_names()
            }
            self._cache[shard_index] = arrays
        return self._cache[shard_index]

    def get_label(self, index: int) -> int:
        shard_index, local = self._locate(index)
        return int(self._arrays(shard_index)["labels"][local])

    def get_packet_count(self, index: int) -> int:
        shard_index, local = self._locate(index)
        key = "packet_mask" if self.is_v3 else "byte_mask"
        return int(self._arrays(shard_index)[key][local].sum())

    def get_capture(self, index: int) -> str:
        shard_index, _ = self._locate(index)
        shard = self.shards[shard_index]
        return str(shard.get("capture", shard["id"]))

    def captures(self) -> np.ndarray:
        return np.asarray(
            [self.get_capture(index) for index in range(len(self))], dtype=np.str_
        )

    def labels(self) -> np.ndarray:
        return np.fromiter((self.get_label(index) for index in range(len(self))), dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, local = self._locate(index)
        arrays = self._arrays(shard_index)
        shard = self.shards[shard_index]
        flow_id = f"{shard['id']}:{local}"
        if self.is_v3:
            return {
                "payload_tokens": torch.as_tensor(
                    np.array(arrays["payload_tokens"][local]), dtype=torch.long
                ),
                "payload_mask": torch.as_tensor(
                    np.array(arrays["payload_mask"][local]), dtype=torch.bool
                ),
                "packet_features": torch.as_tensor(
                    np.array(arrays["packet_features"][local]), dtype=torch.float32
                ),
                "packet_mask": torch.as_tensor(
                    np.array(arrays["packet_mask"][local]), dtype=torch.bool
                ),
                "burst_features": torch.as_tensor(
                    np.array(arrays["burst_features"][local]), dtype=torch.float32
                ),
                "burst_mask": torch.as_tensor(
                    np.array(arrays["burst_mask"][local]), dtype=torch.bool
                ),
                "packet_count": torch.tensor(
                    int(arrays["packet_mask"][local].sum()), dtype=torch.long
                ),
                "label": torch.tensor(int(arrays["labels"][local]), dtype=torch.long),
                "flow_id": flow_id,
                "capture": str(shard.get("capture", shard["id"])),
            }
        return {
            "byte_tokens": torch.as_tensor(np.array(arrays["byte_tokens"][local]), dtype=torch.long),
            "byte_mask": torch.as_tensor(np.array(arrays["byte_mask"][local]), dtype=torch.bool),
            "length_direction": torch.as_tensor(np.array(arrays["length_direction"][local]), dtype=torch.float32),
            "length_mask": torch.as_tensor(np.array(arrays["length_mask"][local]), dtype=torch.bool),
            "packet_count": torch.tensor(int(arrays["byte_mask"][local].sum()), dtype=torch.long),
            "label": torch.tensor(int(arrays["labels"][local]), dtype=torch.long),
            "flow_id": flow_id,
            "capture": str(shard.get("capture", shard["id"])),
        }
