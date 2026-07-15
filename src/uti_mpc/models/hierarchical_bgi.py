from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from uti_mpc.models.pooling import MaskedAttentionPool
from uti_mpc.models.twt import SinusoidalPositionEncoding


class PacketByteConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, groups=dim, bias=False
        )
        self.pointwise = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x).transpose(1, 2)
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        x = self.depthwise(F.pad(x, (left, right))).transpose(1, 2)
        return residual + self.dropout(self.pointwise(F.gelu(x)))


class HierarchicalBGICNN(nn.Module):
    """Encode bytes within each packet, then model the ordered packet sequence."""

    def __init__(
        self,
        byte_embedding_dim: int = 32,
        dim: int = 128,
        heads: int = 4,
        packet_layers: int = 2,
        dropout: float = 0.1,
        max_packets: int = 64,
    ) -> None:
        super().__init__()
        if packet_layers < 1:
            raise ValueError("packet_layers must be positive")
        self.byte_embedding = nn.Embedding(256, byte_embedding_dim)
        self.byte_position = nn.Parameter(torch.zeros(1, 1, 32, byte_embedding_dim))
        field_ids = torch.tensor([0] * 12 + [1] * 16 + [2] * 4, dtype=torch.long)
        self.register_buffer("field_ids", field_ids, persistent=False)
        self.field_embedding = nn.Embedding(3, byte_embedding_dim)
        self.input_projection = nn.Linear(byte_embedding_dim, dim)
        self.byte_convs = nn.ModuleList(
            PacketByteConv(dim, kernel, dropout) for kernel in (3, 5, 7)
        )
        self.byte_mix = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.byte_pool = MaskedAttentionPool(dim)
        self.packet_position = SinusoidalPositionEncoding(dim, max_packets)
        packet_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.packet_encoder = nn.TransformerEncoder(
            packet_layer, num_layers=packet_layers, enable_nested_tensor=False
        )
        self.packet_norm = nn.LayerNorm(dim)
        self.flow_pool = MaskedAttentionPool(dim)
        nn.init.trunc_normal_(self.byte_position, std=0.02)

    def forward(
        self, byte_tokens: torch.Tensor, byte_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if byte_tokens.ndim != 3 or byte_tokens.shape[-1] != 32:
            raise ValueError("byte_tokens must have shape [batch, packets, 32]")
        if byte_tokens.shape[:2] != byte_mask.shape:
            raise ValueError("byte mask does not match byte tokens")
        batch, packets, width = byte_tokens.shape
        embedded = self.byte_embedding(byte_tokens)
        embedded = (
            embedded
            + self.byte_position[:, :, :width]
            + self.field_embedding(self.field_ids[:width])[None, None]
        )
        # Padding rows are removed before any convolution. Packet-independent
        # byte convolutions prevent padding from leaking into valid neighbors.
        embedded = embedded * byte_mask[:, :, None, None].to(embedded.dtype)
        x = self.input_projection(embedded).reshape(batch * packets, width, -1)
        branches = [branch(x) for branch in self.byte_convs]
        x = self.byte_mix(torch.cat(branches, dim=-1))
        byte_positions = torch.ones(
            (batch * packets, width), dtype=torch.bool, device=byte_tokens.device
        )
        packet_tokens, byte_weights = self.byte_pool(x, byte_positions)
        packet_tokens = packet_tokens.reshape(batch, packets, -1)
        packet_tokens = packet_tokens * byte_mask.unsqueeze(-1).to(packet_tokens.dtype)
        packet_tokens = self.packet_position(packet_tokens)
        packet_tokens = packet_tokens.masked_fill(~byte_mask.unsqueeze(-1), 0.0)
        packet_tokens = self.packet_encoder(
            packet_tokens, src_key_padding_mask=~byte_mask
        )
        packet_tokens = self.packet_norm(packet_tokens)
        packet_tokens = packet_tokens.masked_fill(~byte_mask.unsqueeze(-1), 0.0)
        pooled, packet_weights = self.flow_pool(packet_tokens, byte_mask)
        return pooled, packet_tokens, {
            "byte_attention_weights": byte_weights.reshape(batch, packets, width),
            "byte_packet_attention_weights": packet_weights,
        }
