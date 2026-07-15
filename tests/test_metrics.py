import pytest

torch = pytest.importorskip("torch")

from uti_mpc.metrics.open_set import (
    calibrate_open_set,
    class_distance_diagnostics,
    compute_open_set_metrics,
    predict_open_set,
    raw_confusion_matrix,
)


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
