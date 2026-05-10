"""Decoder implementation for the mini Transformer."""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from .attention import MultiHeadAttention
from .encoder import PositionwiseFFN


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_residual: bool = True) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out)) if self.use_residual else self.norm1(self.dropout(self_attn_out))

        cross_attn_out, _ = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(cross_attn_out)) if self.use_residual else self.norm2(self.dropout(cross_attn_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out)) if self.use_residual else self.norm3(self.dropout(ffn_out))
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_residual: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout, use_residual=use_residual) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x
