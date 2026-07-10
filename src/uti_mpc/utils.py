from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def select_single_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"cuda", "cuda:0"}:
        raise ValueError("Only 'cpu' or single logical device 'cuda:0' is supported")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    # Never enumerate, initialize, or wrap additional devices. The physical H200
    # is selected externally with CUDA_VISIBLE_DEVICES; this process uses cuda:0.
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def choose_amp_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    if device.type != "cuda" or requested.lower() == "none":
        return None
    if requested.lower() == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but the selected GPU does not support it")
        return torch.bfloat16
    if requested.lower() == "fp16":
        return torch.float16
    if requested.lower() == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unknown AMP mode: {requested}")


def cosine_warmup_factor(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

