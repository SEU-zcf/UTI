from __future__ import annotations

from typing import Any, Sequence

import torch

from uti_mpc.models.uti_mpc_v3 import UTIMPCV3, predict_v3


def squared_distances(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return torch.cdist(features.float(), prototypes.float(), p=2).square()


def _finite_sample_quantile(values: torch.Tensor, coverage: float) -> torch.Tensor:
    if len(values) == 0:
        raise ValueError("Conformal quantile requires at least one value")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    rank = min(len(values), max(1, int(__import__("math").ceil((len(values) + 1) * coverage))))
    return values.float().sort().values[rank - 1]


@torch.no_grad()
def calibrate_v3_radii(
    model: UTIMPCV3,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    coverage: float = 0.95,
    minimum_subprototype_samples: int = 10,
    minimum_class_samples: int = 20,
) -> dict[str, torch.Tensor]:
    """Calibrate V3 radii using known validation samples only.

    Source codes are 1=subprototype, 2=class fallback, 0=learned-radius fallback.
    """
    device = model.geometry.prototypes.device
    features = validation_features.float().to(device)
    labels = validation_labels.to(device)
    learned = model.geometry.radii().detach()
    calibrated = learned.clone()
    counts = torch.zeros_like(learned, dtype=torch.long)
    source_codes = torch.zeros_like(learned, dtype=torch.long)
    if len(features) == 0:
        return {
            "radii": calibrated.cpu(),
            "learned_radii": learned.cpu(),
            "sample_counts": counts.cpu(),
            "source_codes": source_codes.cpu(),
            "coverage": torch.tensor(coverage),
        }
    distances = model.geometry.distances(features)
    result = predict_v3(model, features)
    nearest_classes = result["nearest_classes"]
    for class_position, label in enumerate(model.geometry.classes):
        correct = (labels == label) & (nearest_classes == label)
        own = distances[:, class_position]
        assignments = (own / learned[class_position].unsqueeze(0)).argmin(dim=1)
        class_values = own[correct].gather(
            1, assignments[correct, None]
        ).squeeze(1)
        class_scale = None
        if len(class_values) >= minimum_class_samples:
            class_ratios = class_values / learned[class_position][assignments[correct]]
            class_scale = _finite_sample_quantile(class_ratios, coverage)
        for prototype in range(model.geometry.subprototypes):
            selected = own[correct & (assignments == prototype), prototype]
            counts[class_position, prototype] = len(selected)
            if len(selected) >= minimum_subprototype_samples:
                calibrated[class_position, prototype] = _finite_sample_quantile(
                    selected, coverage
                ).clamp(model.geometry.min_radius, model.geometry.max_radius)
                source_codes[class_position, prototype] = 1
            elif class_scale is not None:
                calibrated[class_position, prototype] = (
                    learned[class_position, prototype] * class_scale
                ).clamp(model.geometry.min_radius, model.geometry.max_radius)
                source_codes[class_position, prototype] = 2
    return {
        "radii": calibrated.cpu(),
        "learned_radii": learned.cpu(),
        "sample_counts": counts.cpu(),
        "source_codes": source_codes.cpu(),
        "coverage": torch.tensor(coverage),
    }


def _binary_curve(
    scores: torch.Tensor, positives: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = scores.float().cpu()
    positives = positives.bool().cpu()
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_positive = positives[order]
    true_positive = sorted_positive.cumsum(0).float()
    false_positive = (~sorted_positive).cumsum(0).float()
    positive_count = max(int(positives.sum()), 1)
    negative_count = max(int((~positives).sum()), 1)
    distinct = torch.ones(len(scores), dtype=torch.bool)
    if len(scores) > 1:
        distinct[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    return (
        torch.cat((torch.zeros(1), false_positive[distinct] / negative_count)),
        torch.cat((torch.zeros(1), true_positive[distinct] / positive_count)),
        torch.cat((torch.tensor([float("inf")]), sorted_scores[distinct])),
    )


def _macro_f1(
    targets: torch.Tensor, predictions: torch.Tensor, labels: Sequence[int]
) -> float:
    values = []
    for label in labels:
        target = targets == int(label)
        predicted = predictions == int(label)
        tp = float((target & predicted).sum())
        fp = float((~target & predicted).sum())
        fn = float((target & ~predicted).sum())
        denominator = 2.0 * tp + fp + fn
        values.append(2.0 * tp / denominator if denominator else 0.0)
    return sum(values) / len(values) if values else 0.0


def compute_continuous_open_set_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    unknown_scores: torch.Tensor,
    known_classes: Sequence[int],
    unknown_label: int = -1,
) -> dict[str, float]:
    targets = targets.cpu()
    predictions = predictions.cpu()
    unknown_scores = unknown_scores.float().cpu()
    known_mask = torch.zeros_like(targets, dtype=torch.bool)
    for label in known_classes:
        known_mask |= targets == int(label)
    unknown_mask = ~known_mask
    fpr, tpr, _ = _binary_curve(unknown_scores, unknown_mask)
    auroc = float(torch.trapezoid(tpr, fpr)) if unknown_mask.any() and known_mask.any() else 0.0
    order = torch.argsort(unknown_scores, descending=True)
    positives = unknown_mask[order]
    tp = positives.cumsum(0).float()
    precision = tp / torch.arange(1, len(tp) + 1, dtype=torch.float32)
    recall = tp / max(int(unknown_mask.sum()), 1)
    aupr_out = float(
        torch.trapezoid(
            torch.cat((torch.ones(1), precision)),
            torch.cat((torch.zeros(1), recall)),
        )
    ) if unknown_mask.any() else 0.0
    reached = torch.nonzero(tpr >= 0.95, as_tuple=False).flatten()
    fpr95 = float(fpr[reached[0]]) if len(reached) else 1.0

    thresholds = torch.sort(torch.unique(unknown_scores)).values
    oscr_fpr = [0.0]
    oscr_ccr = [0.0]
    nearest_correct = predictions == targets
    for threshold in thresholds:
        accepted = unknown_scores <= threshold
        oscr_fpr.append(float((unknown_mask & accepted).sum()) / max(int(unknown_mask.sum()), 1))
        oscr_ccr.append(float((known_mask & accepted & nearest_correct).sum()) / max(int(known_mask.sum()), 1))
    oscr = float(
        torch.trapezoid(torch.tensor(oscr_ccr), torch.tensor(oscr_fpr))
    ) if unknown_mask.any() and known_mask.any() else 0.0
    grouped_targets = torch.where(
        known_mask, targets, torch.full_like(targets, unknown_label)
    )
    return {
        "AUROC": auroc,
        "AUPR_OUT": aupr_out,
        "FPR95": fpr95,
        "OSCR": oscr,
        "known_macro_F1": _macro_f1(targets, predictions, known_classes),
        "open_macro_F1": _macro_f1(
            grouped_targets,
            predictions,
            [*sorted(int(value) for value in known_classes), unknown_label],
        ),
    }


def class_conditional_knn_scores(
    features: torch.Tensor,
    assigned_classes: torch.Tensor,
    reference_features: torch.Tensor,
    reference_labels: torch.Tensor,
    neighbors: int = 5,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Mean squared distance to the k nearest references of the assigned class."""
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if len(features) != len(assigned_classes):
        raise ValueError("features and assigned_classes must have the same length")
    if len(reference_features) != len(reference_labels):
        raise ValueError("reference_features and reference_labels must have the same length")

    device = features.device
    assigned_classes = assigned_classes.to(device)
    reference_features = reference_features.to(device)
    reference_labels = reference_labels.to(device)
    scores = torch.empty(len(features), dtype=torch.float32, device=device)
    for label in torch.unique(assigned_classes):
        query_indices = torch.nonzero(assigned_classes == label, as_tuple=False).flatten()
        references = reference_features[reference_labels == label]
        if len(references) == 0:
            raise ValueError(f"Assigned class {int(label)} has no kNN reference samples")
        k = min(neighbors, len(references))
        for start in range(0, len(query_indices), chunk_size):
            indices = query_indices[start : start + chunk_size]
            distances = squared_distances(features[indices], references)
            scores[indices] = distances.topk(k, dim=1, largest=False).values.mean(dim=1)
    return scores


def prototype_distance_ratios(distances: torch.Tensor) -> torch.Tensor:
    """Return nearest/second-nearest prototype distance; lower is less ambiguous."""
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError("At least two prototype distances are required")
    nearest_two = distances.topk(2, dim=1, largest=False).values
    epsilon = torch.finfo(nearest_two.dtype).eps
    return nearest_two[:, 0] / nearest_two[:, 1].clamp_min(epsilon)


def calibrate_auxiliary_rejection(
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    validation_distances: torch.Tensor,
    classes: torch.Tensor,
    reference_features: torch.Tensor,
    reference_labels: torch.Tensor,
    quantile: float = 0.95,
    neighbors: int = 5,
    minimum_calibration_samples: int = 5,
    chunk_size: int = 1024,
) -> dict[str, torch.Tensor]:
    """Calibrate local-density and prototype-ambiguity rejection thresholds."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if minimum_calibration_samples < 1:
        raise ValueError("minimum_calibration_samples must be at least 1")
    if len(validation_features) != len(validation_labels):
        raise ValueError("Validation features and labels must have the same length")
    if validation_distances.shape != (len(validation_features), len(classes)):
        raise ValueError("validation_distances has an unexpected shape")

    device = validation_features.device
    classes = classes.to(device)
    validation_labels = validation_labels.to(device)
    nearest_positions = validation_distances.argmin(dim=1)
    nearest_classes = classes[nearest_positions]
    knn_scores = class_conditional_knn_scores(
        validation_features,
        nearest_classes,
        reference_features,
        reference_labels,
        neighbors=neighbors,
        chunk_size=chunk_size,
    )
    margin_ratios = prototype_distance_ratios(validation_distances)
    knn_thresholds = torch.full(
        (len(classes),), torch.inf, dtype=torch.float32, device=device
    )
    margin_thresholds = torch.ones(len(classes), dtype=torch.float32, device=device)
    sample_counts = torch.zeros(len(classes), dtype=torch.long, device=device)
    source_codes = torch.zeros(len(classes), dtype=torch.long, device=device)
    for position, label in enumerate(classes):
        selected = (validation_labels == label) & (nearest_classes == label)
        count = int(selected.sum())
        sample_counts[position] = count
        if count < minimum_calibration_samples:
            continue
        knn_thresholds[position] = torch.quantile(knn_scores[selected], quantile)
        margin_thresholds[position] = torch.quantile(margin_ratios[selected], quantile)
        source_codes[position] = 1
    return {
        "knn_thresholds": knn_thresholds.cpu(),
        "margin_thresholds": margin_thresholds.cpu(),
        "sample_counts": sample_counts.cpu(),
        "source_codes": source_codes.cpu(),
        "quantile": torch.tensor(quantile),
        "neighbors": torch.tensor(neighbors),
    }


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


def compute_subprototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    known_classes: Sequence[int],
    subprototypes_per_class: int = 3,
    iterations: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic spherical k-means prototypes for each known class."""
    if subprototypes_per_class < 2:
        raise ValueError("subprototypes_per_class must be at least 2")
    classes = torch.tensor(sorted(known_classes), dtype=torch.long, device=labels.device)
    class_prototypes = []
    for label in classes:
        selected = torch.nn.functional.normalize(features[labels == label].float(), dim=1)
        if len(selected) < subprototypes_per_class:
            raise ValueError(
                f"Known class {int(label)} has fewer samples than requested subprototypes"
            )
        # Farthest-first initialization is deterministic and avoids seed-sensitive
        # cluster collapse on rare traffic classes.
        first = (selected - selected.mean(dim=0, keepdim=True)).square().sum(dim=1).argmin()
        center_indices = [int(first)]
        minimum_distances = squared_distances(selected, selected[first : first + 1]).squeeze(1)
        for _ in range(1, subprototypes_per_class):
            next_index = int(minimum_distances.argmax())
            center_indices.append(next_index)
            candidate = squared_distances(
                selected, selected[next_index : next_index + 1]
            ).squeeze(1)
            minimum_distances = torch.minimum(minimum_distances, candidate)
        centers = selected[center_indices]
        for _ in range(iterations):
            distances = squared_distances(selected, centers)
            assignments = distances.argmin(dim=1)
            updated = []
            for position in range(subprototypes_per_class):
                members = selected[assignments == position]
                updated.append(
                    torch.nn.functional.normalize(members.mean(dim=0), dim=0)
                    if len(members)
                    else centers[position]
                )
            next_centers = torch.stack(updated)
            if torch.allclose(next_centers, centers, atol=1e-5, rtol=1e-4):
                centers = next_centers
                break
            centers = next_centers
        class_prototypes.append(centers)
    return torch.stack(class_prototypes), classes


def calibrate_subprototype_open_set(
    features: torch.Tensor,
    labels: torch.Tensor,
    known_classes: Sequence[int],
    subprototypes_per_class: int = 3,
    quantile: float = 0.95,
    calibration_features: torch.Tensor | None = None,
    calibration_labels: torch.Tensor | None = None,
    minimum_calibration_samples: int = 5,
    use_train_threshold_floor: bool = True,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    prototypes, classes = compute_subprototypes(
        features, labels, known_classes, subprototypes_per_class
    )
    flat_prototypes = prototypes.flatten(0, 1)
    train_distances = squared_distances(features, flat_prototypes)
    train_thresholds = torch.empty(
        prototypes.shape[:2], dtype=torch.float32, device=features.device
    )
    train_counts = torch.zeros_like(train_thresholds, dtype=torch.long)
    for class_position, label in enumerate(classes):
        class_distances = train_distances[labels == label]
        start = class_position * subprototypes_per_class
        own_distances = class_distances[:, start : start + subprototypes_per_class]
        class_assignments = own_distances.argmin(dim=1)
        class_fallback = torch.quantile(own_distances.amin(dim=1), quantile)
        for subcenter in range(subprototypes_per_class):
            selected = own_distances[class_assignments == subcenter, subcenter]
            train_counts[class_position, subcenter] = len(selected)
            train_thresholds[class_position, subcenter] = (
                torch.quantile(selected, quantile) if len(selected) else class_fallback
            )

    thresholds = train_thresholds.clone()
    sample_counts = train_counts.clone()
    source_codes = torch.zeros_like(train_counts)
    if calibration_features is not None:
        if calibration_labels is None or len(calibration_features) != len(calibration_labels):
            raise ValueError("Held-out calibration features and labels must match")
        calibration_distances = squared_distances(calibration_features, flat_prototypes)
        assignments = calibration_distances.argmin(dim=1)
        class_positions = torch.div(
            assignments, subprototypes_per_class, rounding_mode="floor"
        )
        predicted = classes[class_positions]
        nearest = calibration_distances.gather(1, assignments[:, None]).squeeze(1)
        for class_position, label in enumerate(classes):
            correct_class = (calibration_labels == label) & (predicted == label)
            class_values = nearest[correct_class]
            class_threshold = (
                torch.quantile(class_values, quantile)
                if len(class_values) >= minimum_calibration_samples
                else None
            )
            for subcenter in range(subprototypes_per_class):
                flat_position = class_position * subprototypes_per_class + subcenter
                selected = nearest[correct_class & (assignments == flat_position)]
                sample_counts[class_position, subcenter] = len(selected)
                candidate = None
                code = 0
                if len(selected) >= minimum_calibration_samples:
                    candidate = torch.quantile(selected, quantile)
                    code = 1
                elif class_threshold is not None:
                    candidate = class_threshold
                    code = 2
                if candidate is None:
                    continue
                thresholds[class_position, subcenter] = (
                    torch.maximum(candidate, train_thresholds[class_position, subcenter])
                    if use_train_threshold_floor
                    else candidate
                )
                source_codes[class_position, subcenter] = code
    return {
        "prototypes": prototypes.cpu(),
        "classes": classes.cpu(),
        "thresholds": thresholds.cpu(),
        "train_thresholds": train_thresholds.cpu(),
        "threshold_sample_counts": sample_counts.cpu(),
        "threshold_source_codes": source_codes.cpu(),
        "subprototypes_per_class": torch.tensor(subprototypes_per_class),
        "quantile": torch.tensor(quantile),
    }


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
    if "subprototypes_per_class" in artifacts:
        count = int(artifacts["subprototypes_per_class"])
        distances = squared_distances(features, prototypes.flatten(0, 1))
        nearest_distance, nearest_position = distances.min(dim=1)
        class_positions = torch.div(nearest_position, count, rounding_mode="floor")
        predicted = classes[class_positions]
        selected_thresholds = thresholds.flatten()[nearest_position]
        accepted = nearest_distance <= selected_thresholds
        predicted = torch.where(
            accepted, predicted, torch.full_like(predicted, unknown_label)
        )
        return predicted, nearest_distance
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
