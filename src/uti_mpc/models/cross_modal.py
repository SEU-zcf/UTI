from __future__ import annotations

import torch
from torch import nn

from uti_mpc.models.pooling import MaskedAttentionPool


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        byte_dim: int,
        time_dim: int,
        dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.byte_projection = nn.Linear(byte_dim, dim)
        self.time_projection = nn.Linear(time_dim, dim)
        self.byte_norm = nn.LayerNorm(dim)
        self.time_norm = nn.LayerNorm(dim)
        self.byte_from_time = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.time_from_byte = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.byte_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.time_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.byte_pool = MaskedAttentionPool(dim)
        self.time_pool = MaskedAttentionPool(dim)
        self.reliability_gate = nn.Sequential(
            nn.LayerNorm(dim * 2 + 2),
            nn.Linear(dim * 2 + 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )
        self.output_norm = nn.LayerNorm(dim * 2)

    def forward(
        self,
        byte_tokens: torch.Tensor,
        time_tokens: torch.Tensor,
        byte_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        byte = self.byte_projection(byte_tokens)
        time = self.time_projection(time_tokens)
        byte_context, _ = self.byte_from_time(
            self.byte_norm(byte),
            self.time_norm(time),
            self.time_norm(time),
            key_padding_mask=~time_mask,
            need_weights=False,
        )
        time_context, _ = self.time_from_byte(
            self.time_norm(time),
            self.byte_norm(byte),
            self.byte_norm(byte),
            key_padding_mask=~byte_mask,
            need_weights=False,
        )
        byte = byte + self.dropout(byte_context)
        time = time + self.dropout(time_context)
        byte = byte + self.dropout(self.byte_ffn(byte))
        time = time + self.dropout(self.time_ffn(time))
        byte = byte.masked_fill(~byte_mask.unsqueeze(-1), 0.0)
        time = time.masked_fill(~time_mask.unsqueeze(-1), 0.0)
        byte_pooled, byte_weights = self.byte_pool(byte, byte_mask)
        time_pooled, time_weights = self.time_pool(time, time_mask)
        qualities = torch.stack(
            (
                byte_mask.float().mean(dim=1),
                time_mask.float().mean(dim=1),
            ),
            dim=1,
        ).to(byte_pooled.dtype)
        modality_weights = torch.softmax(
            self.reliability_gate(
                torch.cat((byte_pooled, time_pooled, qualities), dim=1)
            ),
            dim=1,
        )
        fused = torch.cat(
            (
                modality_weights[:, :1] * byte_pooled,
                modality_weights[:, 1:] * time_pooled,
            ),
            dim=1,
        )
        return self.output_norm(fused), {
            "modality_gate": modality_weights,
            "cross_byte_attention_weights": byte_weights,
            "cross_time_attention_weights": time_weights,
        }
