from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    parent = config.pop("extends", None)
    if parent:
        parent_path = (config_path.parent / parent).resolve()
        config = _merge(load_config(parent_path), config)
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, allow_unicode=True, sort_keys=False)


def require_keys(config: dict[str, Any], *keys: str) -> None:
    for dotted in keys:
        current: Any = config
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Missing required configuration key: {dotted}")
            current = current[part]
