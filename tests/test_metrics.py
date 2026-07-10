import pytest

torch = pytest.importorskip("torch")

from uti_mpc.metrics.open_set import calibrate_open_set, compute_open_set_metrics, predict_open_set


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

