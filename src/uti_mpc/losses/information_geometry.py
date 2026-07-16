from __future__ import annotations

from typing import Any
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from uti_mpc.models.uti_mpc_v3 import GeometryHead


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError("Contrastive embeddings and labels must align")
    similarity = embeddings @ embeddings.transpose(0, 1) / temperature
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = (labels[:, None] == labels[None, :]) & ~identity
    logits = similarity - similarity.masked_fill(identity, float("-inf")).amax(
        dim=1, keepdim=True
    ).detach()
    exp_logits = torch.exp(logits).masked_fill(identity, 0.0)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    counts = positive.sum(dim=1)
    valid = counts > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    return -(
        log_probability.masked_fill(~positive, 0.0).sum(dim=1)[valid]
        / counts[valid]
    ).mean()


class InformationGeometryBoundaryLoss(nn.Module):
    def __init__(self, geometry: GeometryHead, config: dict[str, Any]) -> None:
        super().__init__()
        object.__setattr__(self, "_geometry_reference", weakref.ref(geometry))
        self.temperature = float(config.get("contrastive_temperature", 0.1))
        self.softmin_temperature = float(config.get("softmin_temperature", 0.05))
        self.inter_margin = float(config.get("inter_prototype_margin", 0.35))
        self.overlap_gap = float(config.get("boundary_gap", 0.05))
        self.diversity_margin = float(config.get("subprototype_diversity_margin", 0.08))
        self.pseudo_margin = float(config.get("pseudo_unknown_margin", 0.15))
        self.weights = {
            "contrastive": float(config.get("lambda_contrastive", 1.0)),
            "reconstruction": float(config.get("lambda_reconstruction", 0.1)),
            "prototype": float(config.get("lambda_prototype", 1.0)),
            "compact": float(config.get("lambda_compact", 0.5)),
            "separation": float(config.get("lambda_separation", 0.2)),
            "overlap": float(config.get("lambda_overlap", 0.2)),
            "radius": float(config.get("lambda_radius", 0.05)),
            "diversity": float(config.get("lambda_diversity", 0.05)),
            "pseudo_unknown": float(config.get("lambda_pseudo_unknown", 0.5)),
        }

    @property
    def geometry(self) -> GeometryHead:
        geometry = self._geometry_reference()
        if geometry is None:
            raise RuntimeError("V3 geometry head no longer exists")
        return geometry

    @staticmethod
    def _reconstruction(
        details: dict[str, torch.Tensor],
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not mask.any():
            return details["reconstruction"].sum() * 0.0
        return F.smooth_l1_loss(details["reconstruction"][mask], target[mask])

    def _pseudo_unknowns(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        negative_indices = []
        for index, label in enumerate(labels):
            candidates = torch.nonzero(labels != label, as_tuple=False).flatten()
            if len(candidates) == 0:
                return embeddings.sum() * 0.0
            choice = candidates[torch.randint(len(candidates), (), device=labels.device)]
            negative_indices.append(choice)
        negatives = embeddings[torch.stack(negative_indices)]
        mix = torch.distributions.Beta(2.0, 2.0).sample((len(embeddings),)).to(
            embeddings.device, embeddings.dtype
        )
        mix = mix.clamp(0.25, 0.75).unsqueeze(1)
        pseudo = F.normalize(mix * embeddings + (1.0 - mix) * negatives, dim=1)
        best_score = self.geometry.normalized_scores(pseudo).flatten(1).amin(dim=1)
        return F.relu(1.0 + self.pseudo_margin - best_score).mean()

    def _geometry_losses(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        distances = self.geometry.distances(embeddings)
        positions = self.geometry.class_positions(labels)
        own = distances[torch.arange(len(labels), device=labels.device), positions]
        softmin = -self.softmin_temperature * torch.logsumexp(
            -distances / self.softmin_temperature, dim=2
        )
        prototype = F.cross_entropy(-softmin / self.temperature, positions)
        compact = (
            -self.softmin_temperature
            * torch.logsumexp(-own / self.softmin_temperature, dim=1)
        ).mean()

        assigned = own.detach().argmin(dim=1)
        selected_distance = own.gather(1, assigned[:, None]).squeeze(1)
        own_radii = self.geometry.radii()[positions]
        selected_radius = own_radii.gather(1, assigned[:, None]).squeeze(1)
        boundary_cover = F.softplus(
            (selected_distance - selected_radius) / self.softmin_temperature
        ).mean() * self.softmin_temperature
        radius = self.geometry.radii().mean()

        prototypes = self.geometry.normalized_prototypes()
        flat = prototypes.flatten(0, 1)
        center_classes = self.geometry.classes[:, None].expand(
            prototypes.shape[:2]
        ).reshape(-1)
        center_distances = 1.0 - flat @ flat.transpose(0, 1)
        different = center_classes[:, None] != center_classes[None, :]
        separation = F.relu(self.inter_margin - center_distances[different]).mean()
        flat_radii = self.geometry.radii().flatten()
        overlap_values = (
            flat_radii[:, None] + flat_radii[None, :] + self.overlap_gap
            - center_distances
        )
        overlap = F.relu(overlap_values[different]).mean()

        if self.geometry.subprototypes > 1:
            same_distances = 1.0 - torch.einsum("ckd,cjd->ckj", prototypes, prototypes)
            upper = torch.triu(
                torch.ones(
                    self.geometry.subprototypes,
                    self.geometry.subprototypes,
                    dtype=torch.bool,
                    device=embeddings.device,
                ),
                diagonal=1,
            )
            diversity = F.relu(self.diversity_margin - same_distances[:, upper]).mean()
        else:
            diversity = embeddings.sum() * 0.0
        return {
            "prototype": prototype,
            "compact": compact + boundary_cover,
            "separation": separation,
            "overlap": overlap,
            "radius": radius,
            "diversity": diversity,
            "pseudo_unknown": self._pseudo_unknowns(embeddings, labels),
        }

    def forward(
        self,
        first_embedding: torch.Tensor,
        second_embedding: torch.Tensor,
        labels: torch.Tensor,
        first_details: dict[str, torch.Tensor],
        second_details: dict[str, torch.Tensor],
        first_target: torch.Tensor,
        second_target: torch.Tensor,
        first_mask: torch.Tensor,
        second_mask: torch.Tensor,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        joined = torch.cat((first_embedding, second_embedding), dim=0)
        joined_labels = torch.cat((labels, labels), dim=0)
        contrastive = supervised_contrastive_loss(
            joined, joined_labels, self.temperature
        )
        reconstruction = 0.5 * (
            self._reconstruction(first_details, first_target, first_mask)
            + self._reconstruction(second_details, second_target, second_mask)
        )
        zero = joined.sum() * 0.0
        components = {
            "contrastive": contrastive,
            "reconstruction": reconstruction,
            "prototype": zero,
            "compact": zero,
            "separation": zero,
            "overlap": zero,
            "radius": zero,
            "diversity": zero,
            "pseudo_unknown": zero,
        }
        if stage == "formal":
            if not bool(self.geometry.initialized):
                raise RuntimeError("V3 geometry must be initialized before formal training")
            components.update(self._geometry_losses(joined, joined_labels))
        elif stage != "warmup":
            raise ValueError(f"Unknown V3 training stage: {stage}")
        total = sum(self.weights[name] * value for name, value in components.items())
        return {"total": total, **components}
