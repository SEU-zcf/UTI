import pytest

torch = pytest.importorskip("torch")

from uti_mpc.evaluate import _capture_prediction_breakdown


def test_capture_prediction_breakdown_resolves_and_aggregates_shards():
    captures, rows = _capture_prediction_breakdown(
        ["00001_bittorrent:0", "00001_bittorrent:1", "00002_voip:0"],
        {"00001_bittorrent": "NonVPN/bittorrent.pcap", "00002_voip": "NonVPN/voip.pcap"},
        torch.tensor([3, 3, 5]),
        torch.tensor([5, -1, 5]),
        torch.tensor([5, 5, 5]),
        torch.tensor([0.2, 0.4, 0.1]),
        torch.tensor([0.5, 1.2, 0.3]),
    )
    assert captures == ["NonVPN/bittorrent.pcap", "NonVPN/bittorrent.pcap", "NonVPN/voip.pcap"]
    assert len(rows) == 3
    filetransfer_to_voip = next(
        row for row in rows if row["target"] == 3 and row["prediction"] == 5
    )
    assert filetransfer_to_voip["count"] == 1
    assert filetransfer_to_voip["capture"] == "NonVPN/bittorrent.pcap"
