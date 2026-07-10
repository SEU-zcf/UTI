from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from uti_mpc.utils import atomic_torch_save


def rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda0"] = torch.cuda.get_rng_state(0)
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda0" in state:
        torch.cuda.set_rng_state(state["cuda0"], 0)


def save_checkpoint(path: str | Path, payload: dict[str, Any], device: torch.device) -> None:
    payload = {**payload, "rng_state": rng_state(device)}
    atomic_torch_save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)

