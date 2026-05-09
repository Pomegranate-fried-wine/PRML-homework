"""Scaled Dot-Product Attention and Multi-Head Attention modules."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.size()
        x = x.view(bsz, seq_len, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        bsz, n_heads, seq_len, d_head = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, n_heads * d_head)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = self._merge_heads(context)
        output = self.out_proj(context)

        return output, attn
