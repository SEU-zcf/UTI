from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SamePadConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int]) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.kernel_size
        top = (height - 1) // 2
        bottom = height - 1 - top
        left = (width - 1) // 2
        right = width - 1 - left
        return self.conv(F.pad(x, (left, right, top, bottom)))


class ConvBranch(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int]) -> None:
        super().__init__(
            SamePadConv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.network = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.network(x.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
        return x + x * weights, weights.squeeze(-1).squeeze(-1)


class ResidualConvBlock(nn.Module):
    """A lightweight spatial refinement stage that preserves BGI-CNN dimensions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            SamePadConv2d(channels, channels, (3, 3)),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            SamePadConv2d(channels, channels, (3, 3)),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.network(x))


class BGICNN(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 16,
        branch_channels: int = 32,
        output_dim: int = 128,
        se_reduction: int = 8,
        residual_blocks: int = 0,
    ) -> None:
        super().__init__()
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be non-negative")
        self.embedding = nn.Embedding(256, embedding_dim)
        kernels = ((1, 2), (1, 4), (2, 2), (4, 4))
        self.branches = nn.ModuleList(
            ConvBranch(embedding_dim, branch_channels, kernel) for kernel in kernels
        )
        self.mix = nn.Sequential(
            nn.Conv2d(branch_channels * len(kernels), output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )
        self.residual_blocks = nn.Sequential(
            *(ResidualConvBlock(output_dim) for _ in range(residual_blocks))
        )
        self.se = SEBlock(output_dim, se_reduction)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self, byte_tokens: torch.Tensor, byte_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if byte_tokens.ndim != 3 or byte_tokens.shape[-1] != 32:
            raise ValueError("byte_tokens must have shape [batch, packets, 32]")
        x = self.embedding(byte_tokens).permute(0, 3, 1, 2).contiguous()
        x = self.mix(torch.cat([branch(x) for branch in self.branches], dim=1))
        x = self.residual_blocks(x)
        x, se_weights = self.se(x)
        row_mask = byte_mask[:, None, :, None]
        masked = x.masked_fill(~row_mask, torch.finfo(x.dtype).min)
        pooled = masked.amax(dim=(2, 3))
        no_packets = ~byte_mask.any(dim=1)
        if no_packets.any():
            pooled = pooled.masked_fill(no_packets[:, None], 0.0)
        return self.output_norm(pooled), {"se_weights": se_weights}
