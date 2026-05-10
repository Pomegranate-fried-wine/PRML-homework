"""Training entry for Multi30k-based Transformer experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from model.transformer import Transformer
from utils.dataset import TranslationDataset, load_multi30k_pairs
from utils.metrics import token_accuracy
from utils.tokenizer import BasicTokenizer


def make_tgt_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(1, 1, size, size, device=device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=48)
    parser.add_argument("--train_limit", type=int, default=12000)
    parser.add_argument("--valid_limit", type=int, default=1000)
    parser.add_argument("--pe_mode", choices=["sinusoidal", "learned", "none"], default="sinusoidal")
    parser.add_argument("--no_residual", action="store_true")
    parser.add_argument("--save", type=str, default="checkpoints/base.pt")
    parser.add_argument("--log_csv", type=str, default="logs/train_metrics.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pairs = load_multi30k_pairs("train", limit=args.train_limit)
    valid_pairs = load_multi30k_pairs("valid", limit=args.valid_limit)

    src_tok, tgt_tok = BasicTokenizer(min_freq=2), BasicTokenizer(min_freq=2)
    src_tok.fit([p.src for p in train_pairs])
    tgt_tok.fit([p.tgt for p in train_pairs])

    train_loader = DataLoader(TranslationDataset(train_pairs, src_tok, tgt_tok, args.max_len), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TranslationDataset(valid_pairs, src_tok, tgt_tok, args.max_len), batch_size=args.batch_size)

    model = Transformer(len(src_tok.itos), len(tgt_tok.itos), pe_mode=args.pe_mode, use_residual=not args.no_residual).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.stoi[tgt_tok.pad_token])

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "pe_mode", "use_residual"])

    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tr_loss = tr_acc = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            inp, gold = tgt[:, :-1], tgt[:, 1:]
            mask = make_tgt_mask(inp.size(1), device)
            logits = model(src, inp, tgt_mask=mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item(); tr_acc += token_accuracy(logits.detach(), gold, tgt_tok.stoi[tgt_tok.pad_token])

        model.eval()
        va_loss = va_acc = 0.0
        with torch.no_grad():
            for src, tgt in valid_loader:
                src, tgt = src.to(device), tgt.to(device)
                inp, gold = tgt[:, :-1], tgt[:, 1:]
                mask = make_tgt_mask(inp.size(1), device)
                logits = model(src, inp, tgt_mask=mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
                va_loss += loss.item(); va_acc += token_accuracy(logits, gold, tgt_tok.stoi[tgt_tok.pad_token])

        tr_loss /= len(train_loader); tr_acc /= len(train_loader)
        va_loss /= len(valid_loader); va_acc /= len(valid_loader)
        print(f"epoch={epoch+1} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        with open(args.log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, tr_loss, tr_acc, va_loss, va_acc, args.pe_mode, int(not args.no_residual)])

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "src_tok": src_tok.itos, "tgt_tok": tgt_tok.itos, "args": vars(args)}, args.save)
            print(f"saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
