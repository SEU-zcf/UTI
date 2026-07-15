from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from uti_mpc.models.bgi_cnn import BGICNN
from uti_mpc.models.cross_modal import CrossModalFusion
from uti_mpc.models.hierarchical_bgi import HierarchicalBGICNN
from uti_mpc.models.twt import TWT


class UTIMPC(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        byte_dim = int(config.get("byte_dim", 128))
        time_dim = int(config.get("time_dim", 128))
        embedding_dim = int(config.get("embedding_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        self.use_hierarchical_bgi = bool(config.get("hierarchical_bgi", False))
        self.use_cross_modal_fusion = bool(config.get("cross_modal_fusion", False))
        if self.use_cross_modal_fusion and not self.use_hierarchical_bgi:
            raise ValueError("cross_modal_fusion requires hierarchical_bgi")
        if self.use_hierarchical_bgi:
            self.bgi_cnn = HierarchicalBGICNN(
                byte_embedding_dim=int(config.get("byte_embedding_dim", 32)),
                dim=byte_dim,
                heads=int(config.get("byte_attention_heads", 4)),
                packet_layers=int(config.get("byte_packet_layers", 2)),
                dropout=dropout,
                max_packets=int(config.get("max_packets", 64)),
            )
        else:
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
        if self.use_cross_modal_fusion:
            cross_dim = int(config.get("cross_modal_dim", 128))
            self.cross_fusion = CrossModalFusion(
                byte_dim=byte_dim,
                time_dim=time_dim,
                dim=cross_dim,
                heads=int(config.get("cross_attention_heads", 4)),
                dropout=dropout,
            )
            fusion_dim = cross_dim * 2
            self.use_fusion_residual = False
        else:
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
        if self.use_cross_modal_fusion:
            byte_features, byte_packet_tokens, byte_details = self.bgi_cnn(
                byte_tokens, byte_mask
            )
            time_features, time_packet_tokens, time_details = self.twt(
                length_direction, length_mask, return_tokens=True
            )
            fused, fusion_details = self.cross_fusion(
                byte_packet_tokens,
                time_packet_tokens,
                byte_mask,
                length_mask,
            )
            gate = fusion_details["modality_gate"]
        else:
            byte_features, byte_details = self.bgi_cnn(byte_tokens, byte_mask)
            time_features, time_details = self.twt(length_direction, length_mask)
            combined = torch.cat((byte_features, time_features), dim=1)
            gate = torch.sigmoid(self.modality_gate(combined))
            fused = gate * combined
            if self.use_fusion_residual:
                fused = self.fusion_norm(combined + self.fusion_residual(fused))
            fusion_details = {}
        embedding = F.normalize(self.projector(fused), p=2, dim=1, eps=1e-8)
        if not return_details:
            return embedding
        return embedding, {
            "byte_features": byte_features,
            "time_features": time_features,
            "modality_gate": gate,
            **fusion_details,
            **byte_details,
            **time_details,
        }
