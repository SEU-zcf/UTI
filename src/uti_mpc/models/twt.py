from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_length: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_length, dim)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[0]:
            raise ValueError("Sequence exceeds configured positional encoding length")
        return x + self.encoding[: x.shape[1]].to(dtype=x.dtype, device=x.device)


class LocalWindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, dropout: float) -> None:
        super().__init__()
        self.window = window
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        padded_length = math.ceil(length / self.window) * self.window
        padding = padded_length - length
        if padding:
            x = F.pad(x, (0, 0, 0, padding))
            mask = F.pad(mask, (0, padding), value=False)
        windows = padded_length // self.window
        xw = x.reshape(batch, windows, self.window, dim).reshape(-1, self.window, dim)
        mw = mask.reshape(batch, windows, self.window).reshape(-1, self.window)
        key_padding = ~mw
        all_padding = key_padding.all(dim=1)
        if all_padding.any():
            key_padding = key_padding.clone()
            key_padding[all_padding, 0] = False
            xw = xw.clone()
            xw[all_padding, 0] = 0.0
        output, _ = self.attention(xw, xw, xw, key_padding_mask=key_padding, need_weights=False)
        output = output.masked_fill(~mw.unsqueeze(-1), 0.0)
        output = output.reshape(batch, windows, self.window, dim).reshape(batch, padded_length, dim)
        return output[:, :length]


class ScaleGating(nn.Module):
    def __init__(self, dim: int, scales: int) -> None:
        super().__init__()
        hidden = max(16, dim // 4)
        self.scorers = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            for _ in range(scales)
        )

    def forward(
        self, features: list[torch.Tensor], mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(features[0].dtype)
        scores = []
        for feature, scorer in zip(features, self.scorers, strict=True):
            pooled = (feature * mask.unsqueeze(-1)).sum(dim=1) / denominator
            scores.append(scorer(pooled))
        weights = torch.softmax(torch.cat(scores, dim=1), dim=1)
        fused = sum(
            feature * weights[:, index, None, None]
            for index, feature in enumerate(features)
        )
        fused = fused.masked_fill(~mask.unsqueeze(-1), 0.0)
        return fused, weights


class SamePadDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, groups=channels, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        return self.conv(F.pad(x, (left, right)))


class TWTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        windows: tuple[int, ...],
        expansion: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_branches = nn.ModuleList(
            LocalWindowAttention(dim, heads, window, dropout) for window in windows
        )
        self.attention_gate = ScaleGating(dim, len(windows))
        self.attention_norm = nn.LayerNorm(dim)
        hidden = dim * expansion
        self.ffn_expand = nn.Linear(dim, hidden)
        self.conv_branches = nn.ModuleList(
            SamePadDepthwiseConv1d(hidden, kernel) for kernel in windows
        )
        self.ffn_gate = ScaleGating(hidden, len(windows))
        self.ffn_project = nn.Linear(hidden, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attention_features = [branch(x, mask) for branch in self.attention_branches]
        attention, attention_weights = self.attention_gate(attention_features, mask)
        x = self.attention_norm(x + self.dropout(attention))
        hidden = F.gelu(self.ffn_expand(x))
        conv_features = [
            F.gelu(branch(hidden.transpose(1, 2)).transpose(1, 2))
            for branch in self.conv_branches
        ]
        convolved, ffn_weights = self.ffn_gate(conv_features, mask)
        x = self.ffn_norm(x + self.dropout(self.ffn_project(convolved)))
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x, {"attention_scale_weights": attention_weights, "ffn_scale_weights": ffn_weights}


class TWT(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        windows: tuple[int, ...] = (2, 4, 8, 16),
        expansion: int = 4,
        dropout: float = 0.1,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, dim)
        self.position = SinusoidalPositionEncoding(dim, max_length)
        self.block = TWTBlock(dim, heads, windows, expansion, dropout)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self, length_direction: torch.Tensor, length_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.input_projection(length_direction.unsqueeze(-1))
        x = self.position(x)
        x = x.masked_fill(~length_mask.unsqueeze(-1), 0.0)
        x, diagnostics = self.block(x, length_mask)
        denominator = length_mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        pooled = x.sum(dim=1) / denominator
        return self.output_norm(pooled), diagnostics

