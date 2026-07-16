from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import nn

from uti_mpc.models.uti_mpc import UTIMPC
from uti_mpc.models.uti_mpc_v3 import UTIMPCV3


def build_model(config: dict[str, Any], known_classes: Sequence[int]) -> nn.Module:
    name = str(config.get("name", "uti_mpc_v2")).lower()
    if name in {"uti_mpc", "uti_mpc_v1", "uti_mpc_v2"}:
        return UTIMPC(config)
    if name == "uti_mpc_v3":
        return UTIMPCV3(config, known_classes)
    raise ValueError(f"Unknown model.name: {name}")


def is_v3_model(model: nn.Module) -> bool:
    return isinstance(model, UTIMPCV3)

