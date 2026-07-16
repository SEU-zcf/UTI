from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from uti_mpc.models.pooling import MaskedAttentionPool


class PayloadEncoder(nn.Module):
    def __init__(self, width: int, byte_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.width = width
        self.embedding = nn.Embedding(258, byte_dim, padding_idx=256)
        self.position = nn.Parameter(torch.zeros(1, 1, width, byte_dim))
        self.input_projection = nn.Linear(byte_dim, output_dim)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                output_dim,
                output_dim,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=output_dim,
                bias=False,
            )
            for kernel in (3, 5, 7)
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(output_dim * 3),
            nn.Linear(output_dim * 3, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = MaskedAttentionPool(output_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape != mask.shape:
            raise ValueError("payload tokens and mask must have shape [batch, packets, bytes]")
        if tokens.shape[-1] != self.width:
            raise ValueError(f"Expected {self.width} payload bytes, got {tokens.shape[-1]}")
        batch, packets, width = tokens.shape
        safe_tokens = tokens.masked_fill(~mask, 256)
        x = self.embedding(safe_tokens) + self.position[:, :, :width]
        x = self.input_projection(x) * mask.unsqueeze(-1).to(x.dtype)
        flat = x.reshape(batch * packets, width, -1)
        flat_mask = mask.reshape(batch * packets, width)
        branches = []
        for convolution in self.convolutions:
            branch = F.gelu(convolution(flat.transpose(1, 2)).transpose(1, 2))
            branches.append(branch * flat_mask.unsqueeze(-1).to(branch.dtype))
        mixed = self.mix(torch.cat(branches, dim=-1))
        pooled, weights = self.pool(mixed, flat_mask)
        return pooled.reshape(batch, packets, -1), weights.reshape(batch, packets, width)


class GeometryHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        classes: Sequence[int],
        subprototypes: int = 3,
        min_radius: float = 0.03,
        max_radius: float = 0.8,
    ) -> None:
        super().__init__()
        values = sorted(set(int(value) for value in classes))
        if len(values) < 2 or subprototypes < 1:
            raise ValueError("GeometryHead requires at least two classes and one prototype")
        if not 0.0 < min_radius < max_radius < 2.0:
            raise ValueError("Cosine radii must satisfy 0 < min < max < 2")
        self.register_buffer("classes", torch.tensor(values, dtype=torch.long))
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(len(values), subprototypes, embedding_dim), dim=-1)
        )
        self.raw_radii = nn.Parameter(torch.zeros(len(values), subprototypes))
        self.subprototypes = int(subprototypes)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.register_buffer("initialized", torch.tensor(False))

    def normalized_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=-1)

    def radii(self) -> torch.Tensor:
        return self.min_radius + (self.max_radius - self.min_radius) * torch.sigmoid(
            self.raw_radii
        )

    def distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        prototypes = self.normalized_prototypes()
        return 1.0 - torch.einsum("bd,ckd->bck", embeddings, prototypes).clamp(-1.0, 1.0)

    def normalized_scores(
        self, embeddings: torch.Tensor, radii: torch.Tensor | None = None
    ) -> torch.Tensor:
        selected_radii = self.radii() if radii is None else radii.to(embeddings.device)
        return self.distances(embeddings) / selected_radii.clamp_min(1e-6).unsqueeze(0)

    def class_positions(self, labels: torch.Tensor) -> torch.Tensor:
        positions = torch.searchsorted(self.classes, labels)
        safe = positions.clamp_max(len(self.classes) - 1)
        valid = (positions < len(self.classes)) & (self.classes[safe] == labels)
        if not valid.all():
            raise ValueError("Geometry head received a label outside known classes")
        return positions

    @torch.no_grad()
    def initialize_from_embeddings(
        self, embeddings: torch.Tensor, labels: torch.Tensor, iterations: int = 20
    ) -> None:
        embeddings = F.normalize(embeddings.float().to(self.prototypes.device), dim=1)
        labels = labels.to(self.classes.device)
        centers_by_class = []
        radii_by_class = []
        for label in self.classes:
            selected = embeddings[labels == label]
            if len(selected) < self.subprototypes:
                raise ValueError(
                    f"Class {int(label)} has fewer samples than V3 subprototypes"
                )
            first = (selected - selected.mean(dim=0, keepdim=True)).square().sum(1).argmin()
            indices = [int(first)]
            minimum = 1.0 - selected @ selected[first]
            for _ in range(1, self.subprototypes):
                next_index = int(minimum.argmax())
                indices.append(next_index)
                minimum = torch.minimum(
                    minimum, 1.0 - selected @ selected[next_index]
                )
            centers = selected[indices]
            for _ in range(iterations):
                distances = 1.0 - selected @ centers.transpose(0, 1)
                assignments = distances.argmin(1)
                updated = []
                for index in range(self.subprototypes):
                    members = selected[assignments == index]
                    updated.append(
                        F.normalize(members.mean(0), dim=0) if len(members) else centers[index]
                    )
                next_centers = torch.stack(updated)
                if torch.allclose(next_centers, centers, atol=1e-5, rtol=1e-4):
                    centers = next_centers
                    break
                centers = next_centers
            distances = 1.0 - selected @ centers.transpose(0, 1)
            assignments = distances.argmin(1)
            fallback = torch.quantile(distances.amin(1), 0.9)
            class_radii = []
            for index in range(self.subprototypes):
                values = distances[assignments == index, index]
                class_radii.append(torch.quantile(values, 0.9) if len(values) else fallback)
            centers_by_class.append(centers)
            radii_by_class.append(torch.stack(class_radii))
        self.prototypes.copy_(torch.stack(centers_by_class))
        radii = torch.stack(radii_by_class).clamp(
            self.min_radius + 1e-4, self.max_radius - 1e-4
        )
        scaled = (radii - self.min_radius) / (self.max_radius - self.min_radius)
        self.raw_radii.copy_(torch.logit(scaled.clamp(1e-5, 1.0 - 1e-5)))
        self.initialized.fill_(True)


class UTIMPCV3(nn.Module):
    def __init__(self, config: dict[str, Any], known_classes: Sequence[int]) -> None:
        super().__init__()
        packet_dim = int(config.get("packet_dim", 192))
        burst_dim = int(config.get("burst_dim", 128))
        embedding_dim = int(config.get("embedding_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        max_packets = int(config.get("max_packets", 64))
        self.payload_encoder = PayloadEncoder(
            width=int(config.get("payload_bytes", 96)),
            byte_dim=int(config.get("byte_embedding_dim", 32)),
            output_dim=packet_dim,
            dropout=dropout,
        )
        self.packet_features = nn.Sequential(
            nn.LayerNorm(16), nn.Linear(16, packet_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.packet_gate = nn.Linear(packet_dim * 2, packet_dim)
        self.packet_position = nn.Parameter(torch.zeros(1, max_packets, packet_dim))
        self.time_projection = nn.Linear(1, packet_dim, bias=False)
        packet_layer = nn.TransformerEncoderLayer(
            packet_dim,
            int(config.get("packet_heads", 6)),
            packet_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.packet_encoder = nn.TransformerEncoder(
            packet_layer,
            num_layers=int(config.get("packet_layers", 4)),
            enable_nested_tensor=False,
        )
        self.packet_norm = nn.LayerNorm(packet_dim)
        self.packet_pool = MaskedAttentionPool(packet_dim)

        self.burst_projection = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, burst_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.burst_position = nn.Parameter(
            torch.zeros(1, int(config.get("max_bursts", 32)), burst_dim)
        )
        burst_layer = nn.TransformerEncoderLayer(
            burst_dim,
            int(config.get("burst_heads", 4)),
            burst_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.burst_encoder = nn.TransformerEncoder(
            burst_layer,
            num_layers=int(config.get("burst_layers", 2)),
            enable_nested_tensor=False,
        )
        self.burst_to_packet = nn.Linear(burst_dim, packet_dim)
        heads = int(config.get("packet_heads", 6))
        self.packet_from_burst = nn.MultiheadAttention(
            packet_dim, heads, dropout=dropout, batch_first=True
        )
        self.burst_from_packet = nn.MultiheadAttention(
            packet_dim, heads, dropout=dropout, batch_first=True
        )
        self.burst_pool = MaskedAttentionPool(packet_dim)
        self.modality_gate = nn.Sequential(
            nn.LayerNorm(packet_dim * 2),
            nn.Linear(packet_dim * 2, packet_dim),
            nn.GELU(),
            nn.Linear(packet_dim, 2),
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(packet_dim * 2),
            nn.Linear(packet_dim * 2, packet_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(packet_dim, embedding_dim),
        )
        self.reconstruction_head = nn.Linear(packet_dim, 16)
        self.geometry = GeometryHead(
            embedding_dim,
            known_classes,
            subprototypes=int(config.get("subprototypes_per_class", 3)),
            min_radius=float(config.get("minimum_radius", 0.03)),
            max_radius=float(config.get("maximum_radius", 0.8)),
        )
        nn.init.trunc_normal_(self.packet_position, std=0.02)
        nn.init.trunc_normal_(self.burst_position, std=0.02)

    def forward(
        self,
        payload_tokens: torch.Tensor,
        payload_mask: torch.Tensor,
        packet_features: torch.Tensor,
        packet_mask: torch.Tensor,
        burst_features: torch.Tensor,
        burst_mask: torch.Tensor,
        return_details: bool = False,
    ):
        payload, byte_weights = self.payload_encoder(payload_tokens, payload_mask)
        statistics = self.packet_features(packet_features)
        gate = torch.sigmoid(self.packet_gate(torch.cat((payload, statistics), dim=-1)))
        packets = gate * payload + (1.0 - gate) * statistics
        packets = packets + self.packet_position[:, : packets.shape[1]]
        packets = packets + self.time_projection(packet_features[..., 6:7])
        packets = packets.masked_fill(~packet_mask.unsqueeze(-1), 0.0)
        packets = self.packet_encoder(packets, src_key_padding_mask=~packet_mask)
        packets = self.packet_norm(packets).masked_fill(~packet_mask.unsqueeze(-1), 0.0)

        bursts = self.burst_projection(burst_features)
        bursts = bursts + self.burst_position[:, : bursts.shape[1]]
        bursts = self.burst_encoder(bursts, src_key_padding_mask=~burst_mask)
        bursts = self.burst_to_packet(bursts).masked_fill(~burst_mask.unsqueeze(-1), 0.0)
        packet_context, _ = self.packet_from_burst(
            packets, bursts, bursts, key_padding_mask=~burst_mask, need_weights=False
        )
        burst_context, _ = self.burst_from_packet(
            bursts, packets, packets, key_padding_mask=~packet_mask, need_weights=False
        )
        packets = (packets + packet_context).masked_fill(~packet_mask.unsqueeze(-1), 0.0)
        bursts = (bursts + burst_context).masked_fill(~burst_mask.unsqueeze(-1), 0.0)
        packet_pooled, packet_weights = self.packet_pool(packets, packet_mask)
        burst_pooled, burst_weights = self.burst_pool(bursts, burst_mask)
        weights = torch.softmax(
            self.modality_gate(torch.cat((packet_pooled, burst_pooled), dim=-1)), dim=-1
        )
        fused = torch.cat(
            (weights[:, :1] * packet_pooled, weights[:, 1:] * burst_pooled), dim=-1
        )
        embedding = F.normalize(self.projector(fused), dim=1, eps=1e-8)
        if not return_details:
            return embedding
        return embedding, {
            "packet_tokens": packets,
            "reconstruction": self.reconstruction_head(packets),
            "payload_attention": byte_weights,
            "packet_attention": packet_weights,
            "burst_attention": burst_weights,
            "modality_gate": weights,
        }


def predict_v3(
    model: UTIMPCV3,
    embeddings: torch.Tensor,
    calibrated_radii: torch.Tensor | None = None,
    unknown_label: int = -1,
) -> dict[str, torch.Tensor]:
    scores = model.geometry.normalized_scores(embeddings, calibrated_radii)
    flat_scores = scores.flatten(1)
    best_score, best_flat = flat_scores.min(dim=1)
    class_positions = torch.div(
        best_flat, model.geometry.subprototypes, rounding_mode="floor"
    )
    prototype_positions = best_flat % model.geometry.subprototypes
    predictions = model.geometry.classes[class_positions]
    predictions = torch.where(
        best_score <= 1.0,
        predictions,
        torch.full_like(predictions, unknown_label),
    )
    class_scores = scores.amin(dim=2)
    nearest_two = class_scores.topk(2, largest=False, dim=1).values
    return {
        "predictions": predictions,
        "nearest_classes": model.geometry.classes[class_positions],
        "class_positions": class_positions,
        "prototype_positions": prototype_positions,
        "normalized_scores": best_score,
        "second_class_gap": nearest_two[:, 1] - nearest_two[:, 0],
        "distances": model.geometry.distances(embeddings).flatten(1).gather(
            1, best_flat[:, None]
        ).squeeze(1),
    }
