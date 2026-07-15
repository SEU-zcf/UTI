from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from uti_mpc.models.bgi_cnn import BGICNN
from uti_mpc.models.twt import TWT


class UTIMPC(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        byte_dim = int(config.get("byte_dim", 128))
        time_dim = int(config.get("time_dim", 128))
        embedding_dim = int(config.get("embedding_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        self.bgi_cnn = BGICNN(
            embedding_dim=int(config.get("byte_embedding_dim", 16)),
            branch_channels=int(config.get("branch_channels", 32)),
            output_dim=byte_dim,
            se_reduction=int(config.get("se_reduction", 8)),
            residual_blocks=int(config.get("bgi_residual_blocks", 0)),
        )
        self.twt = TWT(
            dim=time_dim,
            heads=int(config.get("attention_heads", 4)),
            windows=tuple(int(value) for value in config.get("windows", [2, 4, 8, 16])),
            expansion=int(config.get("ffn_expansion", 4)),
            dropout=dropout,
            max_length=int(config.get("max_length", 512)),
            depth=int(config.get("twt_depth", 1)),
            shifted_windows=bool(config.get("shifted_windows", False)),
        )
        fusion_dim = byte_dim + time_dim
        self.modality_gate = nn.Linear(fusion_dim, fusion_dim)
        self.use_fusion_residual = bool(config.get("fusion_residual", False))
        if self.use_fusion_residual:
            self.fusion_residual = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, embedding_dim),
        )

    def forward(
        self,
        byte_tokens: torch.Tensor,
        length_direction: torch.Tensor,
        byte_mask: torch.Tensor,
        length_mask: torch.Tensor,
        return_details: bool = False,
    ):
        byte_features, byte_details = self.bgi_cnn(byte_tokens, byte_mask)
        time_features, time_details = self.twt(length_direction, length_mask)
        combined = torch.cat((byte_features, time_features), dim=1)
        gate = torch.sigmoid(self.modality_gate(combined))
        gated = gate * combined
        if self.use_fusion_residual:
            gated = self.fusion_norm(combined + self.fusion_residual(gated))
        embedding = F.normalize(self.projector(gated), p=2, dim=1, eps=1e-8)
        if not return_details:
            return embedding
        return embedding, {
            "byte_features": byte_features,
            "time_features": time_features,
            "modality_gate": gate,
            **byte_details,
            **time_details,
        }
