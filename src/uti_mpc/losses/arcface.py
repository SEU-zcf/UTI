from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ArcFaceLoss(nn.Module):
    """Training-only normalized angular-margin classifier."""

    def __init__(
        self,
        embedding_dim: int,
        classes: Sequence[int],
        scale: float = 30.0,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        class_values = sorted(set(int(label) for label in classes))
        if len(class_values) < 2:
            raise ValueError("ArcFace requires at least two classes")
        if embedding_dim < 1 or scale <= 0.0 or not 0.0 <= margin < math.pi / 2:
            raise ValueError("Invalid ArcFace dimensions, scale, or margin")
        self.register_buffer("classes", torch.tensor(class_values, dtype=torch.long))
        self.weight = nn.Parameter(torch.empty(len(class_values), embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)
        self.cos_margin = math.cos(self.margin)
        self.sin_margin = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.margin_correction = math.sin(math.pi - self.margin) * self.margin

    def _class_positions(self, labels: torch.Tensor) -> torch.Tensor:
        positions = torch.searchsorted(self.classes, labels)
        valid = positions < len(self.classes)
        safe_positions = positions.clamp_max(len(self.classes) - 1)
        valid &= self.classes[safe_positions] == labels
        if not valid.all():
            unexpected = torch.unique(labels[~valid]).tolist()
            raise ValueError(f"ArcFace received labels outside known classes: {unexpected}")
        return positions

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positions = self._class_positions(labels)
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            cosine = F.linear(
                F.normalize(embeddings.float(), dim=1),
                F.normalize(self.weight.float(), dim=1),
            ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
            target_cosine = cosine * self.cos_margin - sine * self.sin_margin
            target_cosine = torch.where(
                cosine > self.threshold,
                target_cosine,
                cosine - self.margin_correction,
            )
            one_hot = F.one_hot(positions, num_classes=len(self.classes)).to(cosine.dtype)
            logits = self.scale * (one_hot * target_cosine + (1.0 - one_hot) * cosine)
            return F.cross_entropy(logits, positions)
