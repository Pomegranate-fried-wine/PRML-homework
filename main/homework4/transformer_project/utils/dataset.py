"""Dataset helpers for toy translation tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class Pair:
    src: str
    tgt: str


class TranslationToyDataset(Dataset):
    def __init__(self, pairs: list[Pair], src_tokenizer, tgt_tokenizer, max_len: int = 64) -> None:
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def _pad(self, ids: list[int], pad_idx: int) -> list[int]:
        ids = ids[: self.max_len]
        return ids + [pad_idx] * (self.max_len - len(ids))

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        pair = self.pairs[idx]
        src = self._pad(self.src_tokenizer.encode(pair.src), self.src_tokenizer.stoi[self.src_tokenizer.pad_token])
        tgt = self._pad(self.tgt_tokenizer.encode(pair.tgt), self.tgt_tokenizer.stoi[self.tgt_tokenizer.pad_token])
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)
