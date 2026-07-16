from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader


V3_INPUT_KEYS = (
    "payload_tokens",
    "payload_mask",
    "packet_features",
    "packet_mask",
    "burst_features",
    "burst_mask",
)


def move_v3_inputs(
    batch: dict, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True) for key in V3_INPUT_KEYS
    }


def forward_batch(
    model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    return_details: bool = False,
):
    if "payload_tokens" in batch:
        inputs = move_v3_inputs(batch, device)
        return model(**inputs, return_details=return_details)
    return model(
        batch["byte_tokens"].to(device, non_blocking=True),
        batch["length_direction"].to(device, non_blocking=True),
        batch["byte_mask"].to(device, non_blocking=True),
        batch["length_mask"].to(device, non_blocking=True),
        return_details=return_details,
    )


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    packet_counts: list[torch.Tensor] = []
    flow_ids: list[str] = []
    for batch in loader:
        with _autocast(device, amp_dtype):
            output = forward_batch(model, batch, device)
        features.append(output.float().cpu())
        labels.append(batch["label"].cpu())
        packet_counts.append(batch["packet_count"].cpu())
        flow_ids.extend(batch["flow_id"])
    if not features:
        raise RuntimeError("Feature loader produced no batches")
    return torch.cat(features), torch.cat(labels), flow_ids, torch.cat(packet_counts)
