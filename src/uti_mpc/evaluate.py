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
    calibrate_auxiliary_rejection,
    calibrate_open_set,
    calibrate_subprototype_open_set,
    calibrate_v3_radii,
    class_conditional_knn_scores,
    class_distance_diagnostics,
    compute_open_set_metrics,
    compute_continuous_open_set_metrics,
    confusion_matrix,
    predict_open_set,
    prototype_distance_ratios,
    raw_confusion_matrix,
    squared_distances,
)
from uti_mpc.models import UTIMPCV3, build_model, predict_v3
from uti_mpc.utils import atomic_torch_save, choose_amp_dtype, seed_everything, select_single_device


def _decision_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    nearest_classes: torch.Tensor,
    known_classes: list[int] | tuple[int, ...],
) -> dict[str, float | int]:
    metrics = compute_open_set_metrics(targets, predictions, known_classes)
    known_mask = torch.zeros_like(targets, dtype=torch.bool)
    for label in known_classes:
        known_mask |= targets == int(label)
    known_count = int(known_mask.sum())
    accepted_known = known_mask & (predictions != -1)
    accepted_known_count = int(accepted_known.sum())
    closed_set_known_correct = int((known_mask & (nearest_classes == targets)).sum())
    metrics.update(
        {
            "closed_set_KCA": closed_set_known_correct / known_count if known_count else 0.0,
            "closed_set_known_correct": closed_set_known_correct,
            "known_test_samples": known_count,
            "known_rejection_rate": float((known_mask & (predictions == -1)).sum())
            / known_count
            if known_count
            else 0.0,
            "accepted_known_accuracy": float(
                (accepted_known & (predictions == targets)).sum()
            )
            / accepted_known_count
            if accepted_known_count
            else 0.0,
        }
    )
    return metrics


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


def _evaluate_v3(
    config: dict,
    checkpoint_path: str | Path,
    model: UTIMPCV3,
    loaders: dict,
    split,
    dataset,
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict:
    if not bool(model.geometry.initialized):
        raise ValueError("V3 checkpoint geometry has not been initialized")
    validation_features, validation_labels, _, _ = extract_embeddings(
        model, loaders["validation"], device, amp_dtype
    )
    test_features, test_labels, flow_ids, packet_counts = extract_embeddings(
        model, loaders["test"], device, amp_dtype
    )
    evaluation = config["evaluation"]
    calibration = calibrate_v3_radii(
        model,
        validation_features,
        validation_labels,
        coverage=float(evaluation.get("coverage", 0.95)),
        minimum_subprototype_samples=int(
            evaluation.get("minimum_subprototype_samples", 10)
        ),
        minimum_class_samples=int(evaluation.get("minimum_class_samples", 20)),
    )
    result = predict_v3(
        model,
        test_features.to(device),
        calibrated_radii=calibration["radii"],
    )
    predictions = result["predictions"].cpu()
    nearest_classes = result["nearest_classes"].cpu()
    unknown_scores = result["normalized_scores"].cpu()
    metrics = {
        **compute_open_set_metrics(test_labels, predictions, split.known_classes),
        **compute_continuous_open_set_metrics(
            test_labels,
            predictions,
            unknown_scores,
            split.known_classes,
        ),
    }
    captures_by_shard = {
        str(shard["id"]): str(shard.get("capture", shard["id"]))
        for shard in dataset.shards
    }
    captures = []
    for flow_id in flow_ids:
        shard_id, separator, _ = flow_id.rpartition(":")
        if not separator or shard_id not in captures_by_shard:
            raise RuntimeError(f"Cannot resolve source capture for flow ID: {flow_id}")
        captures.append(captures_by_shard[shard_id])
    capture_rows = []
    known = set(int(value) for value in split.known_classes)
    for capture in sorted(set(captures)):
        selected = torch.tensor([value == capture for value in captures], dtype=torch.bool)
        capture_targets = test_labels[selected]
        capture_predictions = predictions[selected]
        grouped_targets = torch.tensor(
            [int(value) if int(value) in known else -1 for value in capture_targets]
        )
        capture_metrics = compute_open_set_metrics(
            capture_targets, capture_predictions, split.known_classes
        )
        capture_rows.append(
            {
                "capture": capture,
                "count": int(selected.sum()),
                "open_accuracy": float(
                    (grouped_targets == capture_predictions).float().mean()
                ),
                "KCA": capture_metrics["KCA"],
                "UDR": capture_metrics["UDR"],
            }
        )
    metrics["capture_macro_open_accuracy"] = (
        sum(float(row["open_accuracy"]) for row in capture_rows) / len(capture_rows)
        if capture_rows
        else 0.0
    )
    metrics["checkpoint"] = str(Path(checkpoint_path).resolve())
    metrics["known_classes"] = list(split.known_classes)
    metrics["unknown_classes"] = list(split.unknown_classes)
    metrics["coverage"] = float(calibration["coverage"])
    metrics["thresholds"] = {
        str(int(label)): {
            f"subprototype_{index + 1}": float(radius)
            for index, radius in enumerate(row)
        }
        for label, row in zip(
            model.geometry.classes.cpu(), calibration["radii"], strict=True
        )
    }
    source_names = {0: "learned_radius", 1: "subprototype_validation", 2: "class_validation"}
    metrics["calibration"] = {
        str(int(label)): {
            f"subprototype_{index + 1}": {
                "source": source_names[int(calibration["source_codes"][class_index, index])],
                "samples": int(calibration["sample_counts"][class_index, index]),
            }
            for index in range(model.geometry.subprototypes)
        }
        for class_index, label in enumerate(model.geometry.classes.cpu())
    }
    result_dir = output_dir / "evaluation"
    result_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        **calibration,
        "classes": model.geometry.classes.cpu(),
        "prototypes": model.geometry.normalized_prototypes().detach().cpu(),
    }
    atomic_torch_save(artifacts, result_dir / "open_set_artifacts.pt")
    matrix, order = confusion_matrix(test_labels, predictions, split.known_classes)
    with (result_dir / "confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["target/prediction", *order])
        for label, row in zip(order, matrix.tolist(), strict=True):
            writer.writerow([label, *row])
    with (result_dir / "capture_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["capture", "count", "open_accuracy", "KCA", "UDR"]
        )
        writer.writeheader()
        writer.writerows(capture_rows)
    radii = calibration["radii"]
    learned_radii = calibration["learned_radii"]
    with (result_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "flow_id",
                "capture",
                "packet_count",
                "target",
                "prediction",
                "nearest_class",
                "subprototype",
                "cosine_distance",
                "learned_radius",
                "calibrated_radius",
                "normalized_score",
                "second_class_gap",
            ]
        )
        for index in range(len(flow_ids)):
            class_position = int(result["class_positions"][index])
            prototype_position = int(result["prototype_positions"][index])
            writer.writerow(
                [
                    flow_ids[index],
                    captures[index],
                    int(packet_counts[index]),
                    int(test_labels[index]),
                    int(predictions[index]),
                    int(nearest_classes[index]),
                    prototype_position + 1,
                    float(result["distances"][index]),
                    float(learned_radii[class_position, prototype_position]),
                    float(radii[class_position, prototype_position]),
                    float(unknown_scores[index]),
                    float(result["second_class_gap"][index]),
                ]
            )
    serializable = json.loads(json.dumps(metrics))
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, ensure_ascii=False, indent=2)
    print(json.dumps(serializable, indent=2, ensure_ascii=False))
    return serializable


def evaluate(config_path: str | Path, checkpoint_path: str | Path) -> dict:
    config = load_config(config_path)
    seed_everything(int(config["train"].get("seed", 42)), True)
    output_dir = Path(config["train"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_single_device(str(config["train"].get("device", "cuda:0")))
    amp_dtype = choose_amp_dtype(device, str(config["train"].get("amp", "bf16")))
    dataset, split, _ = load_dataset_and_split(config, output_dir)
    loaders = build_loaders(dataset, split, config)
    model = build_model(config["model"], config["split"]["known_classes"]).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    if isinstance(model, UTIMPCV3):
        return _evaluate_v3(
            config,
            checkpoint_path,
            model,
            loaders,
            split,
            dataset,
            output_dir,
            device,
            amp_dtype,
        )
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
    subprototype_count = int(evaluation_config.get("subprototypes_per_class", 1))
    if subprototype_count > 1:
        if length_conditioned:
            raise ValueError("Subprototypes and length-conditioned prototypes are mutually exclusive")
        artifacts = calibrate_subprototype_open_set(
            train_features,
            train_labels,
            split.known_classes,
            subprototypes_per_class=subprototype_count,
            quantile=float(evaluation_config.get("threshold_quantile", 0.95)),
            calibration_features=validation_features,
            calibration_labels=validation_labels,
            minimum_calibration_samples=int(
                evaluation_config.get("minimum_threshold_samples", 5)
            ),
            use_train_threshold_floor=bool(
                evaluation_config.get("use_train_threshold_floor", True)
            ),
        )
    else:
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
    matrix, order = confusion_matrix(test_labels, predictions, split.known_classes)
    prototypes = artifacts["prototypes"].to(test_features.device)
    prototype_classes = artifacts["classes"].to(test_features.device)
    thresholds = artifacts["thresholds"].to(test_features.device)
    if subprototype_count > 1:
        flat_distances = squared_distances(test_features, prototypes.flatten(0, 1))
        nearest_distances, nearest_prototype_positions = flat_distances.min(dim=1)
        nearest_class_positions = torch.div(
            nearest_prototype_positions, subprototype_count, rounding_mode="floor"
        )
        nearest_classes = prototype_classes[nearest_class_positions]
        nearest_thresholds = thresholds.flatten()[nearest_prototype_positions]
        all_class_distances = flat_distances.reshape(
            len(test_features), len(prototype_classes), subprototype_count
        ).amin(dim=2)
    elif length_conditioned:
        bucket_values = artifacts["bucket_values"].to(test_features.device)
        matches = test_buckets.to(test_features.device)[:, None] == bucket_values[None, :]
        if not matches.any(dim=1).all():
            raise RuntimeError("Evaluation contains a flow-length bucket with no prototype")
        bucket_positions = matches.to(torch.long).argmax(dim=1)
        selected_prototypes = prototypes[:, bucket_positions, :].permute(1, 0, 2)
        all_distances = (test_features[:, None, :] - selected_prototypes).square().sum(dim=2)
        per_sample_thresholds = thresholds[:, bucket_positions].transpose(0, 1)
        nearest_distances, nearest_class_positions = all_distances.min(dim=1)
        nearest_classes = prototype_classes[nearest_class_positions]
        nearest_thresholds = per_sample_thresholds.gather(
            1, nearest_class_positions[:, None]
        ).squeeze(1)
        all_class_distances = all_distances
    else:
        all_distances = squared_distances(test_features, prototypes)
        per_sample_thresholds = thresholds.unsqueeze(0).expand(len(test_features), -1)
        nearest_distances, nearest_class_positions = all_distances.min(dim=1)
        nearest_classes = prototype_classes[nearest_class_positions]
        nearest_thresholds = per_sample_thresholds.gather(
            1, nearest_class_positions[:, None]
        ).squeeze(1)
        all_class_distances = all_distances
    nearest_thresholds = nearest_thresholds.clamp_min(torch.finfo(thresholds.dtype).eps)
    threshold_ratios = nearest_distances / nearest_thresholds
    metrics = _decision_metrics(
        test_labels, predictions, nearest_classes, split.known_classes
    )
    if not torch.allclose(distances, nearest_distances, atol=1e-6, rtol=1e-5):
        raise RuntimeError("Prediction distances do not match nearest prototype distances")

    auxiliary_enabled = bool(
        evaluation_config.get("compare_auxiliary_rejection", False)
    )
    decision_predictions = {"prototype_only": predictions}
    auxiliary_artifacts = None
    test_knn_scores = test_margin_ratios = None
    test_knn_thresholds = test_margin_thresholds = None
    if auxiliary_enabled:
        if validation_features is None or validation_labels is None:
            raise ValueError(
                "Auxiliary rejection comparison requires threshold_source: validation"
            )
        if length_conditioned:
            raise ValueError(
                "Auxiliary rejection comparison currently requires non-conditional prototypes"
            )
        auxiliary_device = device
        classes_on_device = artifacts["classes"].to(auxiliary_device)
        prototypes_on_device = artifacts["prototypes"].to(auxiliary_device)
        validation_features_on_device = validation_features.to(auxiliary_device)
        if subprototype_count > 1:
            validation_distances = squared_distances(
                validation_features_on_device, prototypes_on_device.flatten(0, 1)
            ).reshape(
                len(validation_features), len(classes_on_device), subprototype_count
            ).amin(dim=2)
        else:
            validation_distances = squared_distances(
                validation_features_on_device, prototypes_on_device
            )
        auxiliary_artifacts = calibrate_auxiliary_rejection(
            validation_features_on_device,
            validation_labels.to(auxiliary_device),
            validation_distances,
            classes_on_device,
            train_features.to(auxiliary_device),
            train_labels.to(auxiliary_device),
            quantile=float(
                evaluation_config.get("auxiliary_threshold_quantile", 0.95)
            ),
            neighbors=int(evaluation_config.get("knn_neighbors", 5)),
            minimum_calibration_samples=int(
                evaluation_config.get("minimum_auxiliary_threshold_samples", 5)
            ),
            chunk_size=int(evaluation_config.get("knn_chunk_size", 1024)),
        )
        test_features_on_device = test_features.to(auxiliary_device)
        nearest_classes_on_device = nearest_classes.to(auxiliary_device)
        nearest_positions_on_device = nearest_class_positions.to(auxiliary_device)
        test_knn_scores = class_conditional_knn_scores(
            test_features_on_device,
            nearest_classes_on_device,
            train_features.to(auxiliary_device),
            train_labels.to(auxiliary_device),
            neighbors=int(auxiliary_artifacts["neighbors"]),
            chunk_size=int(evaluation_config.get("knn_chunk_size", 1024)),
        )
        test_margin_ratios = prototype_distance_ratios(
            all_class_distances.to(auxiliary_device)
        )
        test_knn_thresholds = auxiliary_artifacts["knn_thresholds"].to(
            auxiliary_device
        )[nearest_positions_on_device]
        test_margin_thresholds = auxiliary_artifacts["margin_thresholds"].to(
            auxiliary_device
        )[nearest_positions_on_device]
        prototype_accepted = predictions.to(auxiliary_device) != -1
        knn_accepted = prototype_accepted & (test_knn_scores <= test_knn_thresholds)
        margin_accepted = knn_accepted & (
            test_margin_ratios <= test_margin_thresholds
        )
        unknown = torch.full_like(nearest_classes_on_device, -1)
        decision_predictions["prototype_knn"] = torch.where(
            knn_accepted, nearest_classes_on_device, unknown
        ).cpu()
        decision_predictions["prototype_knn_margin"] = torch.where(
            margin_accepted, nearest_classes_on_device, unknown
        ).cpu()
        artifacts["auxiliary_rejection"] = auxiliary_artifacts

    decision_mode_metrics = {
        mode: _decision_metrics(
            test_labels, mode_predictions, nearest_classes, split.known_classes
        )
        for mode, mode_predictions in decision_predictions.items()
    }
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
        if code == 2:
            return "validation_class_fallback"
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
        "subprototypes_per_class": subprototype_count,
        "flow_length_bucket_edges": list(bucket_edges),
        "thresholds": (
            {
                str(int(label)): {
                    f"subprototype_{index + 1}": float(value)
                    for index, value in enumerate(row.tolist())
                }
                for label, row in zip(
                    artifacts["classes"], artifacts["thresholds"], strict=True
                )
            }
            if subprototype_count > 1
            else
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
                    f"subprototype_{index + 1}": int(value)
                    for index, value in enumerate(row.tolist())
                }
                for label, row in zip(artifacts["classes"], sample_counts, strict=True)
            }
            if subprototype_count > 1
            else
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
                    f"subprototype_{index + 1}": source_label(int(value))
                    for index, value in enumerate(row.tolist())
                }
                for label, row in zip(artifacts["classes"], source_codes, strict=True)
            }
            if subprototype_count > 1
            else
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
        "decision_mode_comparison": decision_mode_metrics,
    }
    if auxiliary_artifacts is not None:
        auxiliary_sources = auxiliary_artifacts["source_codes"]

        def finite_float(value: torch.Tensor) -> float | None:
            return float(value) if torch.isfinite(value) else None

        serializable["auxiliary_rejection"] = {
            "threshold_quantile": float(auxiliary_artifacts["quantile"]),
            "knn_neighbors": int(auxiliary_artifacts["neighbors"]),
            "knn_thresholds": {
                str(int(label)): finite_float(value)
                for label, value in zip(
                    artifacts["classes"],
                    auxiliary_artifacts["knn_thresholds"],
                    strict=True,
                )
            },
            "prototype_ratio_thresholds": {
                str(int(label)): finite_float(value)
                for label, value in zip(
                    artifacts["classes"],
                    auxiliary_artifacts["margin_thresholds"],
                    strict=True,
                )
            },
            "calibration_counts": {
                str(int(label)): int(value)
                for label, value in zip(
                    artifacts["classes"],
                    auxiliary_artifacts["sample_counts"],
                    strict=True,
                )
            },
            "sources": {
                str(int(label)): "validation" if int(value) == 1 else "disabled"
                for label, value in zip(
                    artifacts["classes"], auxiliary_sources, strict=True
                )
            },
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
            "closed_set_known_correct",
            "known_test_samples",
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
    with (result_dir / "decision_mode_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "mode",
            "PR",
            "KCA",
            "UDR",
            "closed_set_KCA",
            "closed_set_known_correct",
            "known_test_samples",
            "known_rejection_rate",
            "accepted_known_accuracy",
            "KP",
            "KN",
            "KU",
            "UP",
            "UN",
            "total",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode, mode_metrics in decision_mode_metrics.items():
            writer.writerow({"mode": mode, **mode_metrics})
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
    if auxiliary_artifacts is not None:
        with (result_dir / "auxiliary_predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "flow_id",
                    "target",
                    "nearest_prototype",
                    "prototype_only_prediction",
                    "prototype_knn_prediction",
                    "prototype_knn_margin_prediction",
                    "knn_score",
                    "knn_threshold",
                    "prototype_distance_ratio",
                    "prototype_ratio_threshold",
                ]
            )
            for row in zip(
                flow_ids,
                test_labels.tolist(),
                nearest_classes.tolist(),
                predictions.tolist(),
                decision_predictions["prototype_knn"].tolist(),
                decision_predictions["prototype_knn_margin"].tolist(),
                test_knn_scores.cpu().tolist(),
                test_knn_thresholds.cpu().tolist(),
                test_margin_ratios.cpu().tolist(),
                test_margin_thresholds.cpu().tolist(),
                strict=True,
            ):
                writer.writerow(row)
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
