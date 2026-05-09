"""Training entry for a compact Transformer suitable for RTX 4060."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from model.transformer import Transformer
from utils.dataset import Pair, TranslationToyDataset
from utils.metrics import token_accuracy
from utils.tokenizer import BasicTokenizer


def build_toy_pairs() -> list[Pair]:
    pairs = [
        Pair("i like machine learning", "j aime l apprentissage automatique"),
        Pair("this is a small transformer", "ceci est un petit transformeur"),
        Pair("attention is all you need", "l attention suffit"),
        Pair("we train on tiny data", "nous entrainons sur de petites donnees"),
    ]
    return pairs * 200


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = build_toy_pairs()

    src_tok = BasicTokenizer(min_freq=1)
    tgt_tok = BasicTokenizer(min_freq=1)
    src_tok.fit([p.src for p in pairs])
    tgt_tok.fit([p.tgt for p in pairs])

    dataset = TranslationToyDataset(pairs, src_tok, tgt_tok, max_len=24)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Transformer(len(src_tok.itos), len(tgt_tok.itos)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.stoi[tgt_tok.pad_token])

    for epoch in range(5):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            inp, gold = tgt[:, :-1], tgt[:, 1:]

            logits = model(src, inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_acc += token_accuracy(logits.detach(), gold, tgt_tok.stoi[tgt_tok.pad_token])

        print(f"epoch={epoch+1} loss={total_loss/len(loader):.4f} acc={total_acc/len(loader):.4f}")


if __name__ == "__main__":
    main()
