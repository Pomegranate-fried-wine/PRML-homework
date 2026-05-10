"""Evaluation script for Multi30k checkpoints."""

from __future__ import annotations

import argparse
import json

import torch
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import DataLoader

from model.transformer import Transformer
from utils.dataset import TranslationDataset, load_multi30k_pairs
from utils.tokenizer import BasicTokenizer


def greedy_decode(model, src, bos_id: int, eos_id: int, max_len: int) -> torch.Tensor:
    ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
    for _ in range(max_len - 1):
        out = model(src, ys)
        next_tok = out[:, -1, :].argmax(-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        if (next_tok == eos_id).all():
            break
    return ys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_limit", type=int, default=1000)
    parser.add_argument("--max_len", type=int, default=48)
    parser.add_argument("--save_json", type=str, default="logs/eval_metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    src_tok, tgt_tok = BasicTokenizer(min_freq=1), BasicTokenizer(min_freq=1)
    src_tok.itos, tgt_tok.itos = ckpt["src_tok"], ckpt["tgt_tok"]
    src_tok.stoi = {t: i for i, t in enumerate(src_tok.itos)}
    tgt_tok.stoi = {t: i for i, t in enumerate(tgt_tok.itos)}

    model = Transformer(len(src_tok.itos), len(tgt_tok.itos), pe_mode=ckpt["args"]["pe_mode"], use_residual=not ckpt["args"]["no_residual"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_pairs = load_multi30k_pairs("test", limit=args.test_limit)
    loader = DataLoader(TranslationDataset(test_pairs, src_tok, tgt_tok, args.max_len), batch_size=64)

    refs, hyps = [], []
    examples = []
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            pred = greedy_decode(model, src, tgt_tok.stoi[tgt_tok.bos_token], tgt_tok.stoi[tgt_tok.eos_token], args.max_len)
            for b in range(pred.size(0)):
                hyp = [tgt_tok.itos[i] for i in pred[b].tolist() if i < len(tgt_tok.itos)]
                ref = [tgt_tok.itos[i] for i in tgt[b].tolist() if i < len(tgt_tok.itos)]
                hyp_clean = [w for w in hyp if w not in {tgt_tok.pad_token, tgt_tok.bos_token, tgt_tok.eos_token}]
                ref_clean = [w for w in ref if w not in {tgt_tok.pad_token, tgt_tok.bos_token, tgt_tok.eos_token}]
                hyps.append(hyp_clean)
                refs.append([ref_clean])
                if len(examples) < 10:
                    examples.append({"pred": " ".join(hyp_clean), "ref": " ".join(ref_clean)})

    bleu = corpus_bleu(refs, hyps)
    metrics = {
        "checkpoint": args.ckpt,
        "num_samples": len(hyps),
        "bleu": float(bleu),
        "pe_mode": ckpt["args"]["pe_mode"],
        "use_residual": not ckpt["args"]["no_residual"],
        "examples": examples,
    }
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"BLEU={bleu:.4f} on {len(hyps)} samples")
    print(f"saved metrics to {args.save_json}")


if __name__ == "__main__":
    main()
