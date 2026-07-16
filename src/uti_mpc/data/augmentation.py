from __future__ import annotations

from typing import Any

import torch


def _keep_one(mask: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    empty = ~mask.any(dim=1)
    if empty.any():
        first = original.to(torch.long).argmax(dim=1)
        mask = mask.clone()
        mask[empty, first[empty]] = True
    return mask


def make_v3_view(
    batch: dict[str, Any],
    *,
    packet_drop: float = 0.1,
    payload_mask_fraction: float = 0.1,
    stats_mask_fraction: float = 0.15,
    iat_jitter: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Create a label-preserving augmented V3 view on the current device."""
    payload_tokens = batch["payload_tokens"].clone()
    payload_mask = batch["payload_mask"].clone()
    packet_features = batch["packet_features"].clone()
    packet_mask = batch["packet_mask"].clone()
    burst_features = batch["burst_features"].clone()
    burst_mask = batch["burst_mask"].clone()

    dropped = (torch.rand_like(packet_mask, dtype=torch.float32) < packet_drop) & packet_mask
    kept = _keep_one(packet_mask & ~dropped, packet_mask)
    payload_mask &= kept.unsqueeze(-1)
    packet_features = packet_features.masked_fill(~kept.unsqueeze(-1), 0.0)

    byte_selected = (
        torch.rand_like(payload_mask, dtype=torch.float32) < payload_mask_fraction
    ) & payload_mask
    # 257 is the learned MASK symbol; 256 is reserved for padding inside the encoder.
    payload_tokens[byte_selected] = 257

    reconstruction_mask = (
        torch.rand_like(kept, dtype=torch.float32) < stats_mask_fraction
    ) & kept
    reconstruction_target = packet_features.clone()
    packet_features = packet_features.masked_fill(
        reconstruction_mask.unsqueeze(-1), 0.0
    )
    if iat_jitter > 0.0:
        noise = torch.randn_like(packet_features[..., 5]) * iat_jitter
        packet_features[..., 5] = torch.where(
            kept,
            (packet_features[..., 5] + noise).clamp(0.0, 1.0),
            packet_features[..., 5],
        )

    burst_dropped = (
        torch.rand_like(burst_mask, dtype=torch.float32) < packet_drop * 0.5
    ) & burst_mask
    burst_kept = _keep_one(burst_mask & ~burst_dropped, burst_mask)
    burst_features = burst_features.masked_fill(~burst_kept.unsqueeze(-1), 0.0)
    return {
        "payload_tokens": payload_tokens,
        "payload_mask": payload_mask,
        "packet_features": packet_features,
        "packet_mask": kept,
        "burst_features": burst_features,
        "burst_mask": burst_kept,
        "reconstruction_target": reconstruction_target,
        "reconstruction_mask": reconstruction_mask,
    }

