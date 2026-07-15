from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from uti_mpc.losses.arcface import ArcFaceLoss
from uti_mpc.losses.subcenter import EMALossBalancer, SubcenterPrototypeLoss


def pairwise_squared_distance(embeddings: torch.Tensor) -> torch.Tensor:
    products = embeddings @ embeddings.transpose(0, 1)
    squared_norm = embeddings.square().sum(dim=1, keepdim=True)
    distances = squared_norm + squared_norm.transpose(0, 1) - 2.0 * products
    return distances.clamp_min(0.0)


class ProtoMarginLoss(nn.Module):
    def __init__(
        self,
        triplet_margin: float = 0.2,
        prototype_margin: float = 1.0,
        lambda_intra: float = 0.5,
        lambda_inter: float = 0.3,
        known_classes: list[int] | tuple[int, ...] | None = None,
        embedding_dim: int | None = None,
        lambda_arcface: float = 0.0,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.2,
        subcenters_per_class: int = 1,
        lambda_diversity: float = 0.0,
        subcenter_diversity_margin: float = 0.2,
        loss_weighting: str = "fixed",
        loss_ema_decay: float = 0.95,
    ) -> None:
        super().__init__()
        self.triplet_margin = triplet_margin
        self.prototype_margin = prototype_margin
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter
        self.lambda_arcface = float(lambda_arcface)
        if self.lambda_arcface < 0.0:
            raise ValueError("lambda_arcface cannot be negative")
        self.arcface = (
            ArcFaceLoss(
                int(embedding_dim),
                known_classes,
                scale=arcface_scale,
                margin=arcface_margin,
            )
            if self.lambda_arcface > 0.0
            and embedding_dim is not None
            and known_classes is not None
            else None
        )
        if self.lambda_arcface > 0.0 and self.arcface is None:
            raise ValueError(
                "known_classes and embedding_dim are required when ArcFace is enabled"
            )
        self.lambda_diversity = float(lambda_diversity)
        self.subcenters = (
            SubcenterPrototypeLoss(
                int(embedding_dim),
                known_classes,
                subcenters_per_class=subcenters_per_class,
                inter_margin=prototype_margin,
                diversity_margin=subcenter_diversity_margin,
            )
            if subcenters_per_class > 1
            and embedding_dim is not None
            and known_classes is not None
            else None
        )
        if subcenters_per_class > 1 and self.subcenters is None:
            raise ValueError(
                "known_classes and embedding_dim are required for subcenter prototypes"
            )
        if loss_weighting not in {"fixed", "ema"}:
            raise ValueError("loss_weighting must be 'fixed' or 'ema'")
        self.loss_weighting = loss_weighting
        self.loss_balancer = (
            EMALossBalancer(4, decay=loss_ema_decay) if loss_weighting == "ema" else None
        )

    def _random_triplet(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for anchor in range(len(labels)):
            positive = torch.nonzero(labels == labels[anchor], as_tuple=False).flatten()
            positive = positive[positive != anchor]
            negative = torch.nonzero(labels != labels[anchor], as_tuple=False).flatten()
            if len(positive) == 0 or len(negative) == 0:
                continue
            positive_index = positive[torch.randint(len(positive), (), device=labels.device)]
            negative_index = negative[torch.randint(len(negative), (), device=labels.device)]
            losses.append(
                F.relu(
                    distances[anchor, positive_index]
                    - distances[anchor, negative_index]
                    + self.triplet_margin
                )
            )
        if not losses:
            return distances.sum() * 0.0
        return torch.stack(losses).mean()

    def _batch_hard_triplet(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        same = labels[:, None] == labels[None, :]
        same.fill_diagonal_(False)
        different = ~same
        different.fill_diagonal_(False)
        valid = same.any(dim=1) & different.any(dim=1)
        hardest_positive = distances.masked_fill(~same, float("-inf")).amax(dim=1)
        hardest_negative = distances.masked_fill(~different, float("inf")).amin(dim=1)
        losses = F.relu(hardest_positive - hardest_negative + self.triplet_margin)
        if not valid.any():
            return distances.sum() * 0.0
        return losses[valid].mean()

    def _prototype_losses(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        classes = torch.unique(labels, sorted=True)
        prototypes = torch.stack([embeddings[labels == label].mean(dim=0) for label in classes])
        class_positions = torch.searchsorted(classes, labels)
        assigned = prototypes[class_positions]
        intra = (embeddings - assigned).square().sum(dim=1).mean()
        if len(classes) < 2:
            return intra, embeddings.sum() * 0.0
        distances = torch.cdist(prototypes, prototypes, p=2).square()
        distances.fill_diagonal_(float("inf"))
        nearest = distances.amin(dim=1)
        # Equation (9) in the paper is a sum over batch prototypes.
        inter = F.relu(self.prototype_margin - nearest).sum()
        return intra, inter

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor, stage: str
    ) -> dict[str, torch.Tensor]:
        distances = pairwise_squared_distance(embeddings)
        if stage == "warmup":
            triplet = self._random_triplet(distances, labels)
            zero = triplet.detach() * 0.0
            return {
                "total": triplet,
                "triplet": triplet,
                "intra": zero,
                "inter": zero,
                "arcface": zero,
                "diversity": zero,
            }
        if stage != "formal":
            raise ValueError(f"Unknown training stage: {stage}")
        triplet = self._batch_hard_triplet(distances, labels)
        if self.subcenters is not None:
            intra, inter, diversity = self.subcenters(embeddings, labels)
        else:
            intra, inter = self._prototype_losses(embeddings, labels)
            diversity = embeddings.sum() * 0.0
        arcface = (
            self.arcface(embeddings, labels)
            if self.arcface is not None
            else embeddings.sum() * 0.0
        )
        component_losses = torch.stack((triplet, intra, inter, diversity))
        base_weights = component_losses.new_tensor(
            (1.0, self.lambda_intra, self.lambda_inter, self.lambda_diversity)
        )
        if self.loss_balancer is not None:
            total, effective_weights = self.loss_balancer(
                component_losses, base_weights
            )
        else:
            effective_weights = base_weights
            total = (base_weights * component_losses).sum()
        total = total + self.lambda_arcface * arcface
        result = {
            "total": total,
            "triplet": triplet,
            "intra": intra,
            "inter": inter,
            "arcface": arcface,
            "diversity": diversity,
        }
        for name, weight in zip(
            ("triplet", "intra", "inter", "diversity"),
            effective_weights,
            strict=True,
        ):
            result[f"weight_{name}"] = weight
        return result
