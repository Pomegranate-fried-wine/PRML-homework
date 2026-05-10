"""Visualizations: attention heatmap + training curves + eval comparison bars."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model.transformer import Transformer
from utils.dataset import load_multi30k_pairs
from utils.tokenizer import BasicTokenizer


def plot_training_curves(csv_path: str, out_path: str) -> None:
    epochs, tr_loss, va_loss, tr_acc, va_acc = [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            va_loss.append(float(row["val_loss"]))
            tr_acc.append(float(row["train_acc"]))
            va_acc.append(float(row["val_acc"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, tr_loss, label="train_loss")
    axes[0].plot(epochs, va_loss, label="val_loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[1].plot(epochs, tr_acc, label="train_acc")
    axes[1].plot(epochs, va_acc, label="val_acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)


def plot_eval_bar(eval_json_files: list[str], out_path: str) -> None:
    labels, bleus = [], []
    for fp in eval_json_files:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        labels.append(f"{obj['pe_mode']}-res{int(obj['use_residual'])}")
        bleus.append(obj["bleu"])

    plt.figure(figsize=(8, 4))
    plt.bar(labels, bleus)
    plt.ylabel("BLEU")
    plt.title("Ablation BLEU Comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)


def save_attention_heatmap(ckpt_path: str, index: int, out: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    src_tok, tgt_tok = BasicTokenizer(), BasicTokenizer()
    src_tok.itos, tgt_tok.itos = ckpt["src_tok"], ckpt["tgt_tok"]
    src_tok.stoi = {t: i for i, t in enumerate(src_tok.itos)}
    tgt_tok.stoi = {t: i for i, t in enumerate(tgt_tok.itos)}

    model = Transformer(len(src_tok.itos), len(tgt_tok.itos), pe_mode=ckpt["args"]["pe_mode"], use_residual=not ckpt["args"]["no_residual"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pair = load_multi30k_pairs("test", limit=index + 1)[index]
    src_ids = torch.tensor([src_tok.encode(pair.src)], dtype=torch.long, device=device)
    tgt_ids = torch.tensor([tgt_tok.encode(pair.tgt)[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        src_x = model.dropout(model.pe(model.src_emb(src_ids)))
        tgt_x = model.dropout(model.pe(model.tgt_emb(tgt_ids)))
        memory = model.encoder(src_x)
        x = tgt_x
        attn = None
        for layer in model.decoder.layers:
            _, _ = layer.self_attn(x, x, x)
            cross_out, attn = layer.cross_attn(x, memory, memory)
            x = layer.norm2(x + layer.dropout(cross_out))
            x = layer.norm3(x + layer.dropout(layer.ffn(x)))

    plt.figure(figsize=(8, 6))
    plt.imshow(attn[0, 0].cpu().numpy(), aspect="auto", cmap="viridis")
    plt.title("Cross-attention (head 0)")
    plt.xlabel("Source positions")
    plt.ylabel("Target positions")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["attention", "curves", "bar"], default="attention")
    parser.add_argument("--ckpt", type=str, default="checkpoints/base.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", type=str, default="attention_map.png")
    parser.add_argument("--csv", type=str, default="logs/train_metrics.csv")
    parser.add_argument("--eval_jsons", nargs="*", default=[])
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "attention":
        save_attention_heatmap(args.ckpt, args.index, args.out)
    elif args.mode == "curves":
        plot_training_curves(args.csv, args.out)
    else:
        plot_eval_bar(args.eval_jsons, args.out)

    print(f"saved figure to {args.out}")


if __name__ == "__main__":
    main()
