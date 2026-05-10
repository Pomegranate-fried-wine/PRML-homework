"""Transformer model wiring encoder/decoder, embeddings and positional encoding."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .decoder import Decoder
from .encoder import Encoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, mode: str = "sinusoidal") -> None:
        super().__init__()
        self.mode = mode
        if mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
        elif mode == "learned":
            self.learned = nn.Embedding(max_len, d_model)
        elif mode == "none":
            self.register_buffer("pe", torch.zeros(1, max_len, d_model))
        else:
            raise ValueError(f"Unknown positional encoding mode: {mode}")

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "learned":
            pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
            return x + self.learned(pos)
        return x + self.pe[:, : x.size(1)]


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pe_mode: str = "sinusoidal",
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, mode=pe_mode)
        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout, use_residual=use_residual)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout, use_residual=use_residual)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor | None = None, tgt_mask: Tensor | None = None) -> Tensor:
        src_x = self.dropout(self.pe(self.src_emb(src)))
        tgt_x = self.dropout(self.pe(self.tgt_emb(tgt)))
        memory = self.encoder(src_x, src_mask)
        dec_out = self.decoder(tgt_x, memory, tgt_mask, src_mask)
        return self.generator(dec_out)
