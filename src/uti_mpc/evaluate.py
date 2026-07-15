from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from uti_mpc.config import load_config
from uti_mpc.data.buckets import (
    flow_length_bucket_name,
    packet_counts_to_buckets,
    validate_flow_length_bucket_edges,
)
from uti_mpc.data.labels import ISCXVPN2016_CLASSES
from uti_mpc.engine.checkpoint import load_checkpoint
from uti_mpc.engine.features import extract_embeddings
from uti_mpc.engine.runtime import build_loaders, load_dataset_and_split
from uti_mpc.metrics.open_set import (
    calibrate_open_set,
    class_distance_diagnostics,
    compute_open_set_metrics,
    confusion_matrix,
    predict_open_set,
    raw_confusion_matrix,
    squared_distances,
)
from uti_mpc.models import UTIMPC
from uti_mpc.utils import atomic_torch_save, choose_amp_dtype, seed_everything, select_single_device


def _capture_prediction_breakdown(
    flow_ids: list[str],
    captures_by_shard: dict[str, str],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    nearest_classes: torch.Tensor,
    nearest_distances: torch.Tensor,
    threshold_ratios: torch.Tensor,
) -> tuple[list[str], list[dict[str, float | int | str]]]:
    """Attach flow predictions to source captures and aggregate diagnostic rows."""
    captures: list[str] = []
    grouped: dict[tuple[str, int, int, int], list[tuple[float, float]]] = defaultdict(list)
    for flow_id, target, prediction, nearest, distance, ratio in zip(
        flow_ids,
        targets.tolist(),
        predictions.tolist(),
        nearest_classes.tolist(),
        nearest_distances.tolist(),
        threshold_ratios.tolist(),
        strict=True,
    ):
        shard_id, separator, _ = flow_id.rpartition(":")
        if not separator or shard_id not in captures_by_shard:
            raise RuntimeError(f"Cannot resolve source capture for flow ID: {flow_id}")
        capture = captures_by_shard[shard_id]
        captures.append(capture)
        grouped[(capture, int(target), int(nearest), int(prediction))].append(
            (float(distance), float(ratio))
        )

    rows: list[dict[str, float | int | str]] = []
    for (capture, target, nearest, prediction), values in sorted(grouped.items()):
        distances = torch.tensor([value[0] for value in values], dtype=torch.float32)
        ratios = torch.tensor([value[1] for value in values], dtype=torch.float32)
        rows.append(
            {
                "capture": capture,
                "target": target,
                "target_name": ISCXVPN2016_CLASSES.get(target, "unknown"),
                "nearest_prototype": nearest,
                "prediction": prediction,
                "count": len(values),
                "mean_nearest_squared_distance": float(distances.mean()),
                "median_nearest_squared_distance": float(distances.median()),
                "mean_threshold_ratio": float(ratios.mean()),
                "median_threshold_ratio": float(ratios.median()),
            }
        )
    return captures, rows


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
    train_features, train_labels, _, train_packet_counts = extract_embeddings(
        model, loaders["train_eval"], device, amp_dtype
    )
    evaluation_config = config["evaluation"]
    threshold_source = str(evaluation_config.get("threshold_source", "train")).lower()
    if threshold_source not in {"train", "validation"}:
        raise ValueError("evaluation.threshold_source must be 'train' or 'validation'")
    validation_features = validation_labels = validation_packet_counts = None
    if threshold_source == "validation":
        (
            validation_features,
            validation_labels,
            _,
            validation_packet_counts,
        ) = extract_embeddings(model, loaders["validation"], device, amp_dtype)
    test_features, test_labels, flow_ids, test_packet_counts = extract_embeddings(
        model, loaders["test"], device, amp_dtype
    )
    bucket_edges = validate_flow_length_bucket_edges(
        config["data"].get("flow_length_bucket_edges", [1, 2, 8])
    )
    train_buckets = packet_counts_to_buckets(train_packet_counts, bucket_edges)
    validation_buckets = (
        packet_counts_to_buckets(validation_packet_counts, bucket_edges)
        if validation_packet_counts is not None
        else None
    )
    test_buckets = packet_counts_to_buckets(test_packet_counts, bucket_edges)
    length_conditioned = bool(evaluation_config.get("length_conditioned_prototypes", False))
    artifacts = calibrate_open_set(
        train_features,
        train_labels,
        split.known_classes,
        quantile=float(evaluation_config.get("threshold_quantile", 0.95)),
        buckets=train_buckets if length_conditioned else None,
        bucket_values=list(range(len(bucket_edges) + 1)) if length_conditioned else None,
        calibration_features=validation_features,
        calibration_labels=validation_labels,
        calibration_buckets=validation_buckets if length_conditioned else None,
        minimum_calibration_samples=int(
            evaluation_config.get("minimum_threshold_samples", 5)
        ),
        use_train_threshold_floor=bool(
            evaluation_config.get("use_train_threshold_floor", True)
        ),
    )
    predictions, distances = predict_open_set(
        test_features, artifacts, buckets=test_buckets if length_conditioned else None
    )
    metrics = compute_open_set_metrics(test_labels, predictions, split.known_classes)
    matrix, order = confusion_matrix(test_labels, predictions, split.known_classes)
    prototypes = artifacts["prototypes"].to(test_features.device)
    prototype_classes = artifacts["classes"].to(test_features.device)
    thresholds = artifacts["thresholds"].to(test_features.device)
    if length_conditioned:
        bucket_values = artifacts["bucket_values"].to(test_features.device)
        matches = test_buckets.to(test_features.device)[:, None] == bucket_values[None, :]
        if not matches.any(dim=1).all():
            raise RuntimeError("Evaluation contains a flow-length bucket with no prototype")
        bucket_positions = matches.to(torch.long).argmax(dim=1)
        selected_prototypes = prototypes[:, bucket_positions, :].permute(1, 0, 2)
        all_distances = (test_features[:, None, :] - selected_prototypes).square().sum(dim=2)
        per_sample_thresholds = thresholds[:, bucket_positions].transpose(0, 1)
    else:
        all_distances = squared_distances(test_features, prototypes)
        per_sample_thresholds = thresholds.unsqueeze(0).expand(len(test_features), -1)
    nearest_distances, nearest_positions = all_distances.min(dim=1)
    nearest_classes = prototype_classes[nearest_positions]
    nearest_thresholds = per_sample_thresholds.gather(1, nearest_positions[:, None]).squeeze(1)
    nearest_thresholds = nearest_thresholds.clamp_min(torch.finfo(thresholds.dtype).eps)
    threshold_ratios = nearest_distances / nearest_thresholds
    known_mask = torch.zeros_like(test_labels, dtype=torch.bool)
    for label in split.known_classes:
        known_mask |= test_labels == int(label)
    known_count = int(known_mask.sum())
    accepted_known = known_mask & (predictions != -1)
    accepted_known_count = int(accepted_known.sum())
    closed_set_known_correct = int((known_mask & (nearest_classes == test_labels)).sum())
    metrics.update(
        {
            "closed_set_KCA": closed_set_known_correct / known_count if known_count else 0.0,
            "closed_set_known_correct": closed_set_known_correct,
            "known_test_samples": known_count,
            "known_rejection_rate": float((known_mask & (predictions == -1)).sum()) / known_count
            if known_count
            else 0.0,
            "accepted_known_accuracy": float(
                (accepted_known & (predictions == test_labels)).sum()
            )
            / accepted_known_count
            if accepted_known_count
            else 0.0,
        }
    )
    if not torch.allclose(distances, nearest_distances, atol=1e-6, rtol=1e-5):
        raise RuntimeError("Prediction distances do not match nearest prototype distances")
    raw_target_order = sorted(int(label) for label in torch.unique(test_labels).tolist())
    raw_matrix = raw_confusion_matrix(test_labels, predictions, raw_target_order, order)
    diagnostics = class_distance_diagnostics(
        test_labels,
        predictions,
        nearest_classes,
        nearest_distances,
        threshold_ratios,
        raw_target_order,
        order,
    )
    flow_length_metrics = []
    for bucket in range(len(bucket_edges) + 1):
        selected = test_buckets == bucket
        if not selected.any():
            continue
        flow_length_metrics.append(
            {
                "bucket": bucket,
                "bucket_name": flow_length_bucket_name(bucket, bucket_edges),
                "count": int(selected.sum()),
                **compute_open_set_metrics(
                    test_labels[selected], predictions[selected], split.known_classes
                ),
            }
        )
    # ``capture`` is present in current preprocessing manifests.  The fallback
    # keeps legacy caches and minimal synthetic manifests evaluable.
    captures_by_shard = {
        str(shard["id"]): str(shard.get("capture", shard["id"])) for shard in dataset.shards
    }
    captures, capture_breakdown = _capture_prediction_breakdown(
        flow_ids,
        captures_by_shard,
        test_labels,
        predictions,
        nearest_classes,
        nearest_distances,
        threshold_ratios,
    )
    result_dir = output_dir / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(artifacts, result_dir / "open_set_artifacts.pt")
    source_codes = artifacts["threshold_source_codes"]
    sample_counts = artifacts["threshold_sample_counts"]

    def source_label(code: int) -> str:
        if code == 1:
            return "validation"
        return "train_fallback" if threshold_source == "validation" else "train"

    serializable = {
        **metrics,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "known_classes": list(split.known_classes),
        "unknown_classes": list(split.unknown_classes),
        "threshold_quantile": float(artifacts["quantile"]),
        "threshold_source": threshold_source,
        "minimum_threshold_samples": int(
            evaluation_config.get("minimum_threshold_samples", 5)
        ),
        "use_train_threshold_floor": bool(
            evaluation_config.get("use_train_threshold_floor", True)
        ),
        "length_conditioned_prototypes": length_conditioned,
        "flow_length_bucket_edges": list(bucket_edges),
        "thresholds": (
            {
                str(int(label)): {
                    flow_length_bucket_name(bucket, bucket_edges): float(value)
                    for bucket, value in enumerate(row.tolist())
                }
                for label, row in zip(
                    artifacts["classes"], artifacts["thresholds"], strict=True
                )
            }
            if length_conditioned
            else {
                str(int(label)): float(value)
                for label, value in zip(
                    artifacts["classes"], artifacts["thresholds"], strict=True
                )
            }
        ),
        "threshold_calibration_counts": (
            {
                str(int(label)): {
                    flow_length_bucket_name(bucket, bucket_edges): int(value)
                    for bucket, value in enumerate(row.tolist())
                }
                for label, row in zip(artifacts["classes"], sample_counts, strict=True)
            }
            if length_conditioned
            else {
                str(int(label)): int(value)
                for label, value in zip(artifacts["classes"], sample_counts, strict=True)
            }
        ),
        "threshold_sources": (
            {
                str(int(label)): {
                    flow_length_bucket_name(bucket, bucket_edges): source_label(int(value))
                    for bucket, value in enumerate(row.tolist())
                }
                for label, row in zip(artifacts["classes"], source_codes, strict=True)
            }
            if length_conditioned
            else {
                str(int(label)): source_label(int(value))
                for label, value in zip(artifacts["classes"], source_codes, strict=True)
            }
        ),
        "confusion_order": order,
        "confusion_matrix": matrix.tolist(),
        "raw_target_order": raw_target_order,
        "raw_confusion_matrix": raw_matrix.tolist(),
        "flow_length_metrics": flow_length_metrics,
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, ensure_ascii=False)
    with (result_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key in (
            "PR",
            "KCA",
            "UDR",
            "closed_set_KCA",
            "known_rejection_rate",
            "accepted_known_accuracy",
            "KP",
            "KN",
            "KU",
            "UP",
            "UN",
            "total",
        ):
            writer.writerow([key, metrics[key]])
    with (result_dir / "flow_length_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["bucket", "bucket_name", "count", "PR", "KCA", "UDR", "KP", "KN", "KU", "UP", "UN", "total"],
        )
        writer.writeheader()
        writer.writerows(flow_length_metrics)
    with (result_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target/prediction", *order])
        for label, row in zip(order, matrix.tolist(), strict=True):
            writer.writerow([label, *row])
    with (result_dir / "raw_class_confusion.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target/prediction", *order])
        for label, row in zip(raw_target_order, raw_matrix.tolist(), strict=True):
            writer.writerow([label, *row])
    with (result_dir / "class_distance_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {**entry, "target_name": ISCXVPN2016_CLASSES.get(entry["target"], "unknown")}
                for entry in diagnostics
            ],
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with (result_dir / "capture_prediction_breakdown.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "capture",
                "target",
                "target_name",
                "nearest_prototype",
                "prediction",
                "count",
                "mean_nearest_squared_distance",
                "median_nearest_squared_distance",
                "mean_threshold_ratio",
                "median_threshold_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(capture_breakdown)
    with (result_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "flow_id",
                "capture",
                "packet_count",
                "flow_length_bucket",
                "flow_length_bucket_name",
                "target",
                "prediction",
                "nearest_prototype",
                "nearest_squared_distance",
                "nearest_threshold",
                "threshold_ratio",
            ]
        )
        for flow_id, capture, packet_count, bucket, target, prediction, nearest_class, distance, threshold, ratio in zip(
            flow_ids,
            captures,
            test_packet_counts.tolist(),
            test_buckets.tolist(),
            test_labels.tolist(),
            predictions.tolist(),
            nearest_classes.tolist(),
            nearest_distances.tolist(),
            nearest_thresholds.tolist(),
            threshold_ratios.tolist(),
            strict=True,
        ):
            writer.writerow(
                [
                    flow_id,
                    capture,
                    packet_count,
                    bucket,
                    flow_length_bucket_name(bucket, bucket_edges),
                    target,
                    prediction,
                    nearest_class,
                    distance,
                    threshold,
                    ratio,
                ]
            )
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
