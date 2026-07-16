import pytest

torch = pytest.importorskip("torch")

from uti_mpc.metrics.open_set import (
    calibrate_auxiliary_rejection,
    calibrate_open_set,
    calibrate_subprototype_open_set,
    calibrate_v3_radii,
    class_conditional_knn_scores,
    class_distance_diagnostics,
    compute_open_set_metrics,
    compute_continuous_open_set_metrics,
    predict_open_set,
    prototype_distance_ratios,
    raw_confusion_matrix,
    squared_distances,
)
from uti_mpc.models import UTIMPCV3


def test_calibration_and_open_set_metrics():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([1, 1, 2, 2])
    artifacts = calibrate_open_set(features, labels, [1, 2], quantile=0.95)
    test = torch.tensor([[0.95, 0.05], [0.05, 0.95], [-1.0, -1.0]])
    predictions, _ = predict_open_set(test, artifacts)
    assert predictions[:2].tolist() == [1, 2]
    assert predictions[2].item() == -1
    metrics = compute_open_set_metrics(torch.tensor([1, 2, 3]), predictions, [1, 2])
    assert metrics["PR"] == 1.0
    assert metrics["KCA"] == 1.0
    assert metrics["UDR"] == 1.0


def test_validation_thresholds_keep_train_prototypes_and_record_fallback_metadata():
    train_features = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [0.9, 0.0]]
    )
    train_labels = torch.tensor([1, 1, 2, 2])
    validation_features = torch.tensor([[0.2, 0.0], [0.8, 0.0]])
    validation_labels = torch.tensor([1, 2])

    artifacts = calibrate_open_set(
        train_features,
        train_labels,
        [1, 2],
        quantile=0.95,
        calibration_features=validation_features,
        calibration_labels=validation_labels,
        minimum_calibration_samples=1,
    )

    assert torch.allclose(
        artifacts["prototypes"], torch.tensor([[0.05, 0.0], [0.95, 0.0]])
    )
    assert torch.all(artifacts["thresholds"] > artifacts["train_thresholds"])
    assert artifacts["threshold_sample_counts"].tolist() == [1, 1]
    assert artifacts["threshold_source_codes"].tolist() == [1, 1]

    fallback = calibrate_open_set(
        train_features,
        train_labels,
        [1, 2],
        calibration_features=validation_features,
        calibration_labels=validation_labels,
        minimum_calibration_samples=2,
    )
    assert torch.equal(fallback["thresholds"], fallback["train_thresholds"])
    assert fallback["threshold_source_codes"].tolist() == [0, 0]


def test_auxiliary_rejection_uses_local_class_support_and_prototype_margin():
    references = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]]
    )
    reference_labels = torch.tensor([1, 1, 2, 2])
    prototypes = torch.tensor([[0.05, 0.0], [0.95, 1.0]])
    classes = torch.tensor([1, 2])
    validation = torch.tensor([[0.04, 0.01], [0.96, 0.99]])
    validation_labels = torch.tensor([1, 2])
    validation_distances = squared_distances(validation, prototypes)
    artifacts = calibrate_auxiliary_rejection(
        validation,
        validation_labels,
        validation_distances,
        classes,
        references,
        reference_labels,
        quantile=0.95,
        neighbors=1,
        minimum_calibration_samples=1,
    )
    assert artifacts["source_codes"].tolist() == [1, 1]

    test = torch.tensor([[0.04, 0.01], [0.5, 0.5]])
    distances = squared_distances(test, prototypes)
    nearest_classes = classes[distances.argmin(dim=1)]
    knn_scores = class_conditional_knn_scores(
        test, nearest_classes, references, reference_labels, neighbors=1
    )
    ratios = prototype_distance_ratios(distances)
    threshold_positions = distances.argmin(dim=1)
    assert knn_scores[0] <= artifacts["knn_thresholds"][threshold_positions[0]]
    assert knn_scores[1] > artifacts["knn_thresholds"][threshold_positions[1]]
    assert ratios[0] < ratios[1]


def test_subprototype_calibration_models_multimodal_known_classes():
    features = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.1],
                [1.0, -0.1],
                [0.1, 1.0],
                [-0.1, 1.0],
                [-1.0, 0.1],
                [-1.0, -0.1],
                [0.1, -1.0],
                [-0.1, -1.0],
            ]
        ),
        dim=1,
    )
    labels = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    validation = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.05], [0.05, 1.0], [-1.0, 0.05], [0.05, -1.0]]),
        dim=1,
    )
    validation_labels = torch.tensor([1, 1, 2, 2])
    artifacts = calibrate_subprototype_open_set(
        features,
        labels,
        [1, 2],
        subprototypes_per_class=2,
        calibration_features=validation,
        calibration_labels=validation_labels,
        minimum_calibration_samples=1,
    )
    assert artifacts["prototypes"].shape == (2, 2, 2)
    predictions, _ = predict_open_set(validation, artifacts)
    assert predictions.tolist() == validation_labels.tolist()


def test_raw_confusion_and_distance_diagnostics_preserve_unknown_classes():
    targets = torch.tensor([1, 3, 10, 10])
    predictions = torch.tensor([1, 5, -1, 5])
    nearest = torch.tensor([1, 5, 5, 5])
    distances = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ratios = torch.tensor([0.5, 0.8, 1.2, 1.6])
    order = [1, 5, -1]
    matrix = raw_confusion_matrix(targets, predictions, [1, 3, 10], order)
    assert matrix.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    diagnostics = class_distance_diagnostics(
        targets, predictions, nearest, distances, ratios, [1, 3, 10], order
    )
    unknown_voip = diagnostics[-1]
    assert unknown_voip["target"] == 10
    assert unknown_voip["rejected"] == 1
    assert unknown_voip["nearest_prototype_distribution"]["5"] == 2


def test_continuous_metrics_and_v3_calibration_fallbacks():
    targets = torch.tensor([1, 1, 2, 2, 3, 3])
    predictions = torch.tensor([1, 1, 2, -1, -1, -1])
    scores = torch.tensor([0.1, 0.2, 0.2, 1.1, 1.5, 1.8])
    metrics = compute_continuous_open_set_metrics(
        targets, predictions, scores, [1, 2]
    )
    assert metrics["AUROC"] > 0.9
    assert metrics["OSCR"] > 0.0
    assert 0.0 <= metrics["open_macro_F1"] <= 1.0

    model = UTIMPCV3(
        {
            "payload_bytes": 4,
            "max_packets": 2,
            "max_bursts": 2,
            "byte_embedding_dim": 4,
            "packet_dim": 8,
            "packet_heads": 2,
            "packet_layers": 1,
            "burst_dim": 8,
            "burst_heads": 2,
            "burst_layers": 1,
            "embedding_dim": 2,
            "subprototypes_per_class": 2,
            "dropout": 0.0,
        },
        [1, 2],
    )
    training = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]), dim=1
    )
    model.geometry.initialize_from_embeddings(training, torch.tensor([1, 1, 2, 2]))
    artifacts = calibrate_v3_radii(
        model,
        training[:2],
        torch.tensor([1, 1]),
        minimum_subprototype_samples=1,
        minimum_class_samples=1,
    )
    assert artifacts["radii"].shape == (2, 2)
    assert torch.all(artifacts["source_codes"][1] == 0)
