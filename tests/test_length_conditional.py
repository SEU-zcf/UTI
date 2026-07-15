import pytest

torch = pytest.importorskip("torch")

from uti_mpc.data.buckets import flow_length_bucket_name, packet_counts_to_buckets
from uti_mpc.data.sampler import PKBatchSampler
from uti_mpc.metrics.open_set import calibrate_open_set, predict_open_set


def test_packet_count_buckets_and_stratified_pk_sampling():
    buckets = packet_counts_to_buckets(torch.tensor([1, 2, 3, 8, 9]))
    assert buckets.tolist() == [0, 1, 2, 2, 3]
    assert flow_length_bucket_name(2) == "3-8_packets"

    sampler = PKBatchSampler(
        labels=[1] * 8 + [2] * 8,
        classes_per_batch=2,
        samples_per_class=4,
        buckets=[0, 0, 1, 1, 2, 2, 3, 3] * 2,
        stratify_by_bucket=True,
        batches_per_epoch=1,
    )
    batch = next(iter(sampler))
    sampled_buckets = [([0, 0, 1, 1, 2, 2, 3, 3] * 2)[index] for index in batch]
    for bucket in range(4):
        assert sampled_buckets.count(bucket) == 2


def test_length_conditioned_prototypes_use_matching_flow_length_bucket():
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [-0.9, 0.1], [0.0, -1.0], [0.1, -0.9]]
    )
    labels = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    buckets = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    artifacts = calibrate_open_set(
        features, labels, [1, 2], buckets=buckets, bucket_values=[0, 1]
    )
    assert artifacts["prototypes"].shape == (2, 2, 2)
    predictions, _ = predict_open_set(
        torch.tensor([[0.95, 0.05], [0.05, 0.95]]), artifacts, buckets=torch.tensor([0, 1])
    )
    assert predictions.tolist() == [1, 1]
