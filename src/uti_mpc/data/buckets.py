from __future__ import annotations

from collections.abc import Sequence

import torch


DEFAULT_FLOW_LENGTH_BUCKET_EDGES = (1, 2, 8)


def validate_flow_length_bucket_edges(edges: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(edge) for edge in edges)
    if not normalized or normalized[0] < 1 or any(
        right <= left for left, right in zip(normalized, normalized[1:])
    ):
        raise ValueError("flow-length bucket edges must be strictly increasing positive integers")
    return normalized


def packet_counts_to_buckets(
    packet_counts: torch.Tensor, edges: Sequence[int] = DEFAULT_FLOW_LENGTH_BUCKET_EDGES
) -> torch.Tensor:
    """Map packet counts to contiguous buckets: <=e0, <=e1, ..., >e_last."""
    normalized = validate_flow_length_bucket_edges(edges)
    counts = packet_counts.to(torch.long)
    boundaries = torch.tensor(normalized, dtype=torch.long, device=counts.device)
    return torch.bucketize(counts, boundaries, right=False)


def flow_length_bucket_name(bucket: int, edges: Sequence[int] = DEFAULT_FLOW_LENGTH_BUCKET_EDGES) -> str:
    normalized = validate_flow_length_bucket_edges(edges)
    if not 0 <= bucket <= len(normalized):
        raise ValueError(f"Unknown flow-length bucket {bucket}")
    if bucket == 0:
        return f"1-{normalized[0]}_packets"
    if bucket == len(normalized):
        return f"{normalized[-1] + 1}_plus_packets"
    return f"{normalized[bucket - 1] + 1}-{normalized[bucket]}_packets"
