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


def calibrate_open_set(
    features: torch.Tensor,
    labels: torch.Tensor,
    known_classes: Sequence[int],
    quantile: float = 0.95,
) -> dict[str, torch.Tensor]:
    prototypes, classes = compute_prototypes(features, labels, known_classes)
    distances = squared_distances(features, prototypes)
    nearest_positions = distances.argmin(dim=1)
    predicted = classes[nearest_positions]
    thresholds = []
    for position, label in enumerate(classes):
        class_mask = labels == label
        correctly_classified = class_mask & (predicted == label)
        selected = distances[correctly_classified, position]
        if len(selected) == 0:
            selected = distances[class_mask, position]
        thresholds.append(torch.quantile(selected.float(), quantile))
    return {
        "prototypes": prototypes.cpu(),
        "classes": classes.cpu(),
        "thresholds": torch.stack(thresholds).cpu(),
        "quantile": torch.tensor(quantile),
    }


def predict_open_set(
    features: torch.Tensor,
    artifacts: dict[str, torch.Tensor],
    unknown_label: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    prototypes = artifacts["prototypes"].to(features.device)
    classes = artifacts["classes"].to(features.device)
    thresholds = artifacts["thresholds"].to(features.device)
    distances = squared_distances(features, prototypes)
    nearest_distance, nearest_position = distances.min(dim=1)
    predicted = classes[nearest_position]
    accepted = nearest_distance <= thresholds[nearest_position]
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
