from __future__ import annotations

import torch
from torch import nn


class MaskedAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(16, dim // 2)
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[:2] != mask.shape:
            raise ValueError("Token and mask dimensions do not match")
        logits = self.score(tokens).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        no_tokens = ~mask.any(dim=1)
        if no_tokens.any():
            logits = logits.clone()
            logits[no_tokens, 0] = 0.0
        weights = torch.softmax(logits, dim=1).masked_fill(~mask, 0.0)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled.masked_fill(no_tokens.unsqueeze(-1), 0.0)
        return pooled, weights
