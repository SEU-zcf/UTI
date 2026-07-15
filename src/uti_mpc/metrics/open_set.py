from __future__ import annotations

from typing import Any, Sequence

import torch


def squared_distances(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return torch.cdist(features.float(), prototypes.float(), p=2).square()


def compute_prototypes(
    features: torch.Tensor, labels: torch.Tensor, known_classes: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    class_tensor = torch.tensor(sorted(known_classes), dtype=torch.long, device=labels.device)
    prototypes = []
    for label in class_tensor:
        selected = features[labels == label]
        if len(selected) == 0:
            raise ValueError(f"Known class {int(label)} has no calibration samples")
        prototypes.append(selected.mean(dim=0))
    return torch.stack(prototypes), class_tensor


def _bucket_positions(assignments: torch.Tensor, bucket_values: torch.Tensor) -> torch.Tensor:
    matches = assignments.to(bucket_values.device)[:, None] == bucket_values[None, :]
    if not matches.any(dim=1).all():
        missing = torch.unique(assignments[~matches.any(dim=1)]).tolist()
        raise ValueError(f"No prototype bucket is available for assignments: {missing}")
    return matches.to(torch.long).argmax(dim=1)


def _conditional_squared_distances(
    features: torch.Tensor, prototypes: torch.Tensor, bucket_positions: torch.Tensor
) -> torch.Tensor:
    # prototypes: [classes, buckets, dimensions]. Select a class prototype for
    # each sample's observed flow-length bucket, producing [samples, classes].
    selected = prototypes[:, bucket_positions, :].permute(1, 0, 2)
    return (features[:, None, :] - selected).square().sum(dim=2)


def calibrate_open_set(
    features: torch.Tensor,
    labels: torch.Tensor,
    known_classes: Sequence[int],
    quantile: float = 0.95,
    buckets: torch.Tensor | None = None,
    bucket_values: Sequence[int] | None = None,
    calibration_features: torch.Tensor | None = None,
    calibration_labels: torch.Tensor | None = None,
    calibration_buckets: torch.Tensor | None = None,
    minimum_calibration_samples: int = 5,
    use_train_threshold_floor: bool = True,
) -> dict[str, torch.Tensor]:
    """Build train prototypes and calibrate class-specific rejection thresholds.

    ``features`` always defines the prototypes.  When ``calibration_features``
    is supplied, thresholds are estimated from correctly classified held-out
    known samples.  Classes (or class/bucket pairs) with too few held-out
    samples fall back to their train-derived threshold.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if minimum_calibration_samples < 1:
        raise ValueError("minimum_calibration_samples must be at least 1")
    if (calibration_features is None) != (calibration_labels is None):
        raise ValueError(
            "calibration_features and calibration_labels must be provided together"
        )

    prototypes, classes = compute_prototypes(features, labels, known_classes)
    if calibration_features is not None:
        if (
            calibration_features.ndim != features.ndim
            or calibration_features.shape[1:] != features.shape[1:]
        ):
            raise ValueError("Calibration features do not match prototype feature dimensions")
        if len(calibration_features) != len(calibration_labels):
            raise ValueError("Calibration features and labels must have the same length")
        known = set(int(label) for label in classes.tolist())
        unexpected = sorted(set(int(label) for label in calibration_labels.tolist()) - known)
        if unexpected:
            raise ValueError(f"Threshold calibration contains unknown classes: {unexpected}")

    if buckets is not None:
        if len(buckets) != len(labels):
            raise ValueError("buckets must have the same length as labels")
        values = torch.tensor(
            sorted(set(int(value) for value in (bucket_values or torch.unique(buckets).tolist()))),
            dtype=torch.long,
            device=features.device,
        )
        buckets_on_device = buckets.to(features.device)
        positions = _bucket_positions(buckets_on_device, values)
        conditional_prototypes = torch.empty(
            (len(classes), len(values), features.shape[1]), dtype=features.dtype, device=features.device
        )
        for class_position, label in enumerate(classes):
            class_features = features[labels == label]
            for bucket_position, bucket in enumerate(values):
                selected = class_features[buckets_on_device[labels == label] == bucket]
                conditional_prototypes[class_position, bucket_position] = (
                    selected.mean(dim=0) if len(selected) else prototypes[class_position]
                )

        global_distances = squared_distances(features, prototypes)
        global_predicted = classes[global_distances.argmin(dim=1)]
        global_thresholds = []
        for position, label in enumerate(classes):
            class_mask = labels == label
            selected = global_distances[class_mask & (global_predicted == label), position]
            if len(selected) == 0:
                selected = global_distances[class_mask, position]
            global_thresholds.append(torch.quantile(selected.float(), quantile))

        distances = _conditional_squared_distances(features, conditional_prototypes, positions)
        predicted = classes[distances.argmin(dim=1)]
        thresholds = torch.empty(
            (len(classes), len(values)), dtype=torch.float32, device=features.device
        )
        threshold_counts = torch.zeros_like(thresholds, dtype=torch.long)
        for class_position, label in enumerate(classes):
            for bucket_position, bucket in enumerate(values):
                class_mask = (labels == label) & (buckets_on_device == bucket)
                selected = distances[class_mask & (predicted == label), class_position]
                if len(selected) == 0:
                    selected = distances[class_mask, class_position]
                threshold_counts[class_position, bucket_position] = len(selected)
                thresholds[class_position, bucket_position] = (
                    torch.quantile(selected.float(), quantile)
                    if len(selected)
                    else global_thresholds[class_position]
                )

        train_thresholds = thresholds.clone()
        threshold_source_codes = torch.zeros_like(thresholds, dtype=torch.long)
        if calibration_features is not None:
            if calibration_buckets is None or len(calibration_buckets) != len(
                calibration_labels
            ):
                raise ValueError(
                    "calibration_buckets matching calibration_labels are required for conditional calibration"
                )
            calibration_positions = _bucket_positions(
                calibration_buckets.to(features.device), values
            )
            calibration_distances = _conditional_squared_distances(
                calibration_features.to(features.device),
                conditional_prototypes,
                calibration_positions,
            )
            calibration_predicted = classes[calibration_distances.argmin(dim=1)]
            calibration_labels_on_device = calibration_labels.to(features.device)
            calibration_buckets_on_device = calibration_buckets.to(features.device)
            for class_position, label in enumerate(classes):
                for bucket_position, bucket in enumerate(values):
                    selected = calibration_distances[
                        (calibration_labels_on_device == label)
                        & (calibration_buckets_on_device == bucket)
                        & (calibration_predicted == label),
                        class_position,
                    ]
                    threshold_counts[class_position, bucket_position] = len(selected)
                    if len(selected) < minimum_calibration_samples:
                        continue
                    candidate = torch.quantile(selected.float(), quantile)
                    thresholds[class_position, bucket_position] = (
                        torch.maximum(
                            candidate, train_thresholds[class_position, bucket_position]
                        )
                        if use_train_threshold_floor
                        else candidate
                    )
                    threshold_source_codes[class_position, bucket_position] = 1
        return {
            "prototypes": conditional_prototypes.cpu(),
            "classes": classes.cpu(),
            "thresholds": thresholds.cpu(),
            "train_thresholds": train_thresholds.cpu(),
            "threshold_sample_counts": threshold_counts.cpu(),
            "threshold_source_codes": threshold_source_codes.cpu(),
            "bucket_values": values.cpu(),
            "quantile": torch.tensor(quantile),
        }

    distances = squared_distances(features, prototypes)
    nearest_positions = distances.argmin(dim=1)
    predicted = classes[nearest_positions]
    thresholds = []
    train_counts = []
    for position, label in enumerate(classes):
        class_mask = labels == label
        correctly_classified = class_mask & (predicted == label)
        selected = distances[correctly_classified, position]
        if len(selected) == 0:
            selected = distances[class_mask, position]
        train_counts.append(len(selected))
        thresholds.append(torch.quantile(selected.float(), quantile))
    train_thresholds = torch.stack(thresholds)
    threshold_counts = torch.tensor(train_counts, dtype=torch.long, device=features.device)
    threshold_source_codes = torch.zeros(
        len(classes), dtype=torch.long, device=features.device
    )

    if calibration_features is not None:
        calibration_features_on_device = calibration_features.to(features.device)
        calibration_labels_on_device = calibration_labels.to(features.device)
        calibration_distances = squared_distances(calibration_features_on_device, prototypes)
        calibration_predicted = classes[calibration_distances.argmin(dim=1)]
        thresholds = train_thresholds.clone()
        for position, label in enumerate(classes):
            selected = calibration_distances[
                (calibration_labels_on_device == label)
                & (calibration_predicted == label),
                position,
            ]
            threshold_counts[position] = len(selected)
            if len(selected) < minimum_calibration_samples:
                continue
            candidate = torch.quantile(selected.float(), quantile)
            thresholds[position] = (
                torch.maximum(candidate, train_thresholds[position])
                if use_train_threshold_floor
                else candidate
            )
            threshold_source_codes[position] = 1
    else:
        thresholds = train_thresholds

    return {
        "prototypes": prototypes.cpu(),
        "classes": classes.cpu(),
        "thresholds": thresholds.cpu(),
        "train_thresholds": train_thresholds.cpu(),
        "threshold_sample_counts": threshold_counts.cpu(),
        "threshold_source_codes": threshold_source_codes.cpu(),
        "quantile": torch.tensor(quantile),
    }


def predict_open_set(
    features: torch.Tensor,
    artifacts: dict[str, torch.Tensor],
    unknown_label: int = -1,
    buckets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = artifacts["prototypes"].to(features.device)
    classes = artifacts["classes"].to(features.device)
    thresholds = artifacts["thresholds"].to(features.device)
    if "bucket_values" in artifacts:
        if buckets is None:
            raise ValueError("Flow-length buckets are required by conditional prototypes")
        bucket_values = artifacts["bucket_values"].to(features.device)
        positions = _bucket_positions(buckets.to(features.device), bucket_values)
        distances = _conditional_squared_distances(features, prototypes, positions)
        per_sample_thresholds = thresholds[:, positions].transpose(0, 1)
    else:
        distances = squared_distances(features, prototypes)
        per_sample_thresholds = thresholds.unsqueeze(0).expand(len(features), -1)
    nearest_distance, nearest_position = distances.min(dim=1)
    predicted = classes[nearest_position]
    accepted = nearest_distance <= per_sample_thresholds.gather(1, nearest_position[:, None]).squeeze(1)
    predicted = torch.where(accepted, predicted, torch.full_like(predicted, unknown_label))
    return predicted, nearest_distance


def compute_open_set_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    known_classes: Sequence[int],
    unknown_label: int = -1,
) -> dict[str, float | int]:
    targets = targets.cpu()
    predictions = predictions.cpu()
    known_mask = torch.zeros_like(targets, dtype=torch.bool)
    for label in known_classes:
        known_mask |= targets == int(label)
    unknown_mask = ~known_mask
    kp = int((known_mask & (predictions == targets)).sum())
    ku = int((known_mask & (predictions == unknown_label)).sum())
    kn = int(known_mask.sum()) - kp - ku
    up = int((unknown_mask & (predictions == unknown_label)).sum())
    un = int(unknown_mask.sum()) - up
    total = kp + kn + ku + up + un
    return {
        "PR": (kp + up) / total if total else 0.0,
        "KCA": kp / (kp + kn + ku) if (kp + kn + ku) else 0.0,
        "UDR": up / (up + un) if (up + un) else 0.0,
        "KP": kp,
        "KN": kn,
        "KU": ku,
        "UP": up,
        "UN": un,
        "total": total,
    }


def confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    known_classes: Sequence[int],
    unknown_label: int = -1,
) -> tuple[torch.Tensor, list[int]]:
    order = [int(label) for label in sorted(known_classes)] + [unknown_label]
    mapping = {label: index for index, label in enumerate(order)}
    matrix = torch.zeros((len(order), len(order)), dtype=torch.long)
    known = set(known_classes)
    for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
        target_group = target if target in known else unknown_label
        prediction_group = prediction if prediction in known else unknown_label
        matrix[mapping[target_group], mapping[prediction_group]] += 1
    return matrix, order


def raw_confusion_matrix(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    target_order: Sequence[int],
    prediction_order: Sequence[int],
) -> torch.Tensor:
    """Confusion matrix that preserves original labels, including unknown classes."""
    target_index = {int(label): index for index, label in enumerate(target_order)}
    prediction_index = {int(label): index for index, label in enumerate(prediction_order)}
    matrix = torch.zeros((len(target_order), len(prediction_order)), dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist(), strict=True):
        if target not in target_index:
            raise ValueError(f"Target label {target} is not in target_order")
        if prediction not in prediction_index:
            raise ValueError(f"Prediction label {prediction} is not in prediction_order")
        matrix[target_index[target], prediction_index[prediction]] += 1
    return matrix


def class_distance_diagnostics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    nearest_classes: torch.Tensor,
    nearest_distances: torch.Tensor,
    threshold_ratios: torch.Tensor,
    target_order: Sequence[int],
    prediction_order: Sequence[int],
    unknown_label: int = -1,
) -> list[dict[str, Any]]:
    """Summarize rejection and distance behavior for each original target class."""
    quantiles = (0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    result: list[dict[str, Any]] = []
    for target in target_order:
        selected = targets == int(target)
        count = int(selected.sum())
        if count == 0:
            continue
        class_predictions = predictions[selected]
        class_nearest = nearest_classes[selected]
        distances = nearest_distances[selected].float()
        ratios = threshold_ratios[selected].float()
        distance_quantiles = torch.quantile(distances, torch.tensor(quantiles)).tolist()
        ratio_quantiles = torch.quantile(ratios, torch.tensor(quantiles)).tolist()
        result.append(
            {
                "target": int(target),
                "count": count,
                "accepted": int((class_predictions != unknown_label).sum()),
                "rejected": int((class_predictions == unknown_label).sum()),
                "rejection_rate": float((class_predictions == unknown_label).float().mean()),
                "prediction_distribution": {
                    str(label): int((class_predictions == int(label)).sum())
                    for label in prediction_order
                },
                "nearest_prototype_distribution": {
                    str(label): int((class_nearest == int(label)).sum())
                    for label in prediction_order
                    if label != unknown_label
                },
                "nearest_distance_quantiles": {
                    str(quantile): float(value)
                    for quantile, value in zip(quantiles, distance_quantiles, strict=True)
                },
                "threshold_ratio_quantiles": {
                    str(quantile): float(value)
                    for quantile, value in zip(quantiles, ratio_quantiles, strict=True)
                },
            }
        )
    return result
