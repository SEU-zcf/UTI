from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from uti_mpc.config import load_config
from uti_mpc.engine.checkpoint import load_checkpoint
from uti_mpc.engine.features import extract_embeddings
from uti_mpc.engine.runtime import build_loaders, load_dataset_and_split
from uti_mpc.metrics.open_set import (
    calibrate_open_set,
    compute_open_set_metrics,
    confusion_matrix,
    predict_open_set,
)
from uti_mpc.models import UTIMPC
from uti_mpc.utils import atomic_torch_save, choose_amp_dtype, seed_everything, select_single_device


def evaluate(config_path: str | Path, checkpoint_path: str | Path) -> dict:
    config = load_config(config_path)
    seed_everything(int(config["train"].get("seed", 42)), True)
    output_dir = Path(config["train"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_single_device(str(config["train"].get("device", "cuda:0")))
    amp_dtype = choose_amp_dtype(device, str(config["train"].get("amp", "bf16")))
    dataset, split, _ = load_dataset_and_split(config, output_dir)
    loaders = build_loaders(dataset, split, config)
    model = UTIMPC(config["model"]).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    train_features, train_labels, _ = extract_embeddings(
        model, loaders["train_eval"], device, amp_dtype
    )
    test_features, test_labels, flow_ids = extract_embeddings(
        model, loaders["test"], device, amp_dtype
    )
    artifacts = calibrate_open_set(
        train_features,
        train_labels,
        split.known_classes,
        quantile=float(config["evaluation"].get("threshold_quantile", 0.95)),
    )
    predictions, distances = predict_open_set(test_features, artifacts)
    metrics = compute_open_set_metrics(test_labels, predictions, split.known_classes)
    matrix, order = confusion_matrix(test_labels, predictions, split.known_classes)
    result_dir = output_dir / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(artifacts, result_dir / "open_set_artifacts.pt")
    serializable = {
        **metrics,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "known_classes": list(split.known_classes),
        "unknown_classes": list(split.unknown_classes),
        "threshold_quantile": float(artifacts["quantile"]),
        "thresholds": {
            str(int(label)): float(value)
            for label, value in zip(artifacts["classes"], artifacts["thresholds"], strict=True)
        },
        "confusion_order": order,
        "confusion_matrix": matrix.tolist(),
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)
    with (result_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in ("PR", "KCA", "UDR", "KP", "KN", "KU", "UP", "UN", "total"):
            writer.writerow([key, metrics[key]])
    with (result_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target/prediction", *order])
        for label, row in zip(order, matrix.tolist(), strict=True):
            writer.writerow([label, *row])
    with (result_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["flow_id", "target", "prediction", "nearest_squared_distance"])
        for flow_id, target, prediction, distance in zip(
            flow_ids, test_labels.tolist(), predictions.tolist(), distances.tolist(), strict=True
        ):
            writer.writerow([flow_id, target, prediction, distance])
    print(json.dumps(serializable, indent=2, ensure_ascii=False))
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate UTI-MPC")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()

