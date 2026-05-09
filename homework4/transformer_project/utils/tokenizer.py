"""Simple whitespace tokenizer for fast local experiments."""

from __future__ import annotations

from collections import Counter


class BasicTokenizer:
    def __init__(self, min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.stoi = {}
        self.itos = []

    def fit(self, texts: list[str]) -> None:
        counter = Counter(token for text in texts for token in text.strip().split())
        self.itos = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for token, freq in counter.items():
            if freq >= self.min_freq:
                self.itos.append(token)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        ids = [self.stoi[self.bos_token]]
        for tok in text.strip().split():
            ids.append(self.stoi.get(tok, self.stoi[self.unk_token]))
        ids.append(self.stoi[self.eos_token])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i < len(self.itos))
