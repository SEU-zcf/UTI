from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader


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
            output = model(
                batch["byte_tokens"].to(device, non_blocking=True),
                batch["length_direction"].to(device, non_blocking=True),
                batch["byte_mask"].to(device, non_blocking=True),
                batch["length_mask"].to(device, non_blocking=True),
            )
        features.append(output.float().cpu())
        labels.append(batch["label"].cpu())
        packet_counts.append(batch["packet_count"].cpu())
        flow_ids.extend(batch["flow_id"])
    if not features:
        raise RuntimeError("Feature loader produced no batches")
    return torch.cat(features), torch.cat(labels), flow_ids, torch.cat(packet_counts)
