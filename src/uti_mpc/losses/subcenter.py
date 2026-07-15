from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class SubcenterPrototypeLoss(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        classes: Sequence[int],
        subcenters_per_class: int = 3,
        inter_margin: float = 1.0,
        diversity_margin: float = 0.2,
    ) -> None:
        super().__init__()
        class_values = sorted(set(int(label) for label in classes))
        if len(class_values) < 2 or subcenters_per_class < 2:
            raise ValueError("Subcenter loss requires at least two classes and two subcenters")
        self.register_buffer("classes", torch.tensor(class_values, dtype=torch.long))
        self.centers = nn.Parameter(
            torch.empty(len(class_values), subcenters_per_class, embedding_dim)
        )
        nn.init.normal_(self.centers, std=0.02)
        self.inter_margin = float(inter_margin)
        self.diversity_margin = float(diversity_margin)

    def _class_positions(self, labels: torch.Tensor) -> torch.Tensor:
        positions = torch.searchsorted(self.classes, labels)
        valid = positions < len(self.classes)
        safe = positions.clamp_max(len(self.classes) - 1)
        valid &= self.classes[safe] == labels
        if not valid.all():
            raise ValueError("Subcenter loss received labels outside known classes")
        return positions

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = F.normalize(self.centers, dim=-1)
        positions = self._class_positions(labels)
        assigned_centers = centers[positions]
        within = (embeddings[:, None, :] - assigned_centers).square().sum(dim=-1)
        intra = within.amin(dim=1).mean()

        flat = centers.flatten(0, 1)
        center_classes = self.classes[:, None].expand(centers.shape[:2]).reshape(-1)
        distances = torch.cdist(flat, flat, p=2).square()
        different = center_classes[:, None] != center_classes[None, :]
        nearest_other = distances.masked_fill(~different, float("inf")).amin(dim=1)
        inter = F.relu(self.inter_margin - nearest_other).mean()

        if centers.shape[1] < 2:
            diversity = embeddings.sum() * 0.0
        else:
            same_class_distances = torch.cdist(centers, centers, p=2).square()
            pair_mask = torch.triu(
                torch.ones(
                    centers.shape[1],
                    centers.shape[1],
                    dtype=torch.bool,
                    device=centers.device,
                ),
                diagonal=1,
            )
            diversity = F.relu(
                self.diversity_margin - same_class_distances[:, pair_mask]
            ).mean()
        return intra, inter, diversity


class EMALossBalancer(nn.Module):
    """Normalize loss coefficients using their moving numerical scales."""

    def __init__(self, components: int, decay: float = 0.95, epsilon: float = 1e-4) -> None:
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.register_buffer("ema", torch.ones(components))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(
        self, losses: torch.Tensor, base_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        detached = losses.detach().float().clamp_min(self.epsilon)
        with torch.no_grad():
            if not bool(self.initialized):
                self.ema.copy_(detached)
                self.initialized.fill_(True)
            else:
                self.ema.mul_(self.decay).add_(detached, alpha=1.0 - self.decay)
        active = (base_weights > 0) & (losses.detach().float() > self.epsilon)
        raw_weights = torch.where(
            active,
            base_weights.float() / self.ema.clamp_min(self.epsilon),
            torch.zeros_like(base_weights, dtype=torch.float32),
        )
        if not active.any():
            return losses.sum() * 0.0, raw_weights
        effective = raw_weights / raw_weights.sum().clamp_min(self.epsilon)
        effective = effective * base_weights[active].sum()
        return (effective.to(losses.dtype) * losses).sum(), effective.detach()
