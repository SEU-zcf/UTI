from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from uti_mpc.config import load_config
from uti_mpc.evaluate import evaluate
from uti_mpc.train import train


METRICS = (
    "PR",
    "KCA",
    "UDR",
    "AUROC",
    "AUPR_OUT",
    "FPR95",
    "OSCR",
    "known_macro_F1",
    "open_macro_F1",
    "capture_macro_open_accuracy",
)


def benchmark(config_paths: list[str | Path], output: str | Path) -> dict:
    rows = []
    for config_path in config_paths:
        checkpoint = train(config_path)
        result = evaluate(config_path, checkpoint)
        config = load_config(config_path)
        rows.append(
            {
                "config": str(Path(config_path).resolve()),
                "seed": int(config["train"].get("seed", 42)),
                **{metric: float(result[metric]) for metric in METRICS},
            }
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {}
    for metric in METRICS:
        values = torch.tensor([row[metric] for row in rows], dtype=torch.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=len(values) > 1)),
        }
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "seed", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    payload = {"runs": rows, "summary": summary}
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential UTI-MPC experiments")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", default="outputs/iscxvpn2016_ur20_v3_benchmark")
    args = parser.parse_args()
    benchmark(args.configs, args.output)


if __name__ == "__main__":
    main()

