from __future__ import annotations

from typing import Sequence

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

