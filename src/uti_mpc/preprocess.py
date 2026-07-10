from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from uti_mpc.config import load_config, require_keys
from uti_mpc.data.flow import iter_capture_flows
from uti_mpc.data.labels import ISCXVPN2016_CLASSES, LabelResolver


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:80]


def preprocess(
    config_path: str | Path,
    pcap_root: str | Path,
    output_dir: str | Path | None = None,
    label_map: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    require_keys(config, "data.cache_dir", "data.np", "data.nl", "data.idle_timeout")
    root = Path(pcap_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PCAP root does not exist: {root}")
    destination = Path(output_dir or config["data"]["cache_dir"]).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    captures = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}
    )
    if not captures:
        raise FileNotFoundError(f"No .pcap or .pcapng files found under {root}")
    resolver = LabelResolver(root, label_map)
    data_config = config["data"]
    manifest: dict = {
        "version": 1,
        "dataset": "ISCXVPN2016",
        "source_root": str(root),
        "class_map": {str(key): value for key, value in ISCXVPN2016_CLASSES.items()},
        "feature_config": {
            "np": int(data_config["np"]),
            "nl": int(data_config["nl"]),
            "idle_timeout": float(data_config["idle_timeout"]),
            "min_packets": int(data_config.get("min_packets", 1)),
            "length_mode": "ipv4_total_length",
        },
        "shards": [],
    }
    total = 0
    for capture_index, capture in enumerate(tqdm(captures, desc="PCAP files")):
        label = resolver.resolve(capture)
        flows = list(
            iter_capture_flows(
                capture,
                npackets=int(data_config["np"]),
                nlengths=int(data_config["nl"]),
                idle_timeout=float(data_config["idle_timeout"]),
                min_packets=int(data_config.get("min_packets", 1)),
            )
        )
        if not flows:
            continue
        shard_id = f"{capture_index:05d}_{_safe_stem(capture)}"
        shard_dir = destination / "shards" / shard_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "byte_tokens": np.stack([flow.byte_tokens for flow in flows]),
            "byte_mask": np.stack([flow.byte_mask for flow in flows]),
            "length_direction": np.stack([flow.length_direction for flow in flows]),
            "length_mask": np.stack([flow.length_mask for flow in flows]),
            "labels": np.full(len(flows), label, dtype=np.int64),
        }
        paths: dict[str, str] = {}
        for name, array in arrays.items():
            array_path = shard_dir / f"{name}.npy"
            np.save(array_path, array, allow_pickle=False)
            paths[name] = array_path.relative_to(destination).as_posix()
        metadata_path = shard_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "capture": capture.relative_to(root).as_posix(),
                    "label": label,
                    "flow_ids": [flow.flow_id for flow in flows],
                },
                handle,
                ensure_ascii=False,
            )
        manifest["shards"].append(
            {
                "id": shard_id,
                "capture": capture.relative_to(root).as_posix(),
                "label": label,
                "count": len(flows),
                "metadata": metadata_path.relative_to(destination).as_posix(),
                **paths,
            }
        )
        total += len(flows)
    if not manifest["shards"]:
        raise RuntimeError("No valid IPv4 TCP/UDP flows were extracted")
    manifest["total_samples"] = total
    manifest_path = destination / "manifest.json"
    temporary = destination / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temporary.replace(manifest_path)
    print(f"Wrote {total} flows in {len(manifest['shards'])} shards to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ISCXVPN2016 PCAP files for UTI-MPC")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pcap-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--label-map")
    args = parser.parse_args()
    preprocess(args.config, args.pcap_root, args.output_dir, args.label_map)


if __name__ == "__main__":
    main()

