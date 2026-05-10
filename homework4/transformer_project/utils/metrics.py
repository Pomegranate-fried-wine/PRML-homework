"""Simple token-level metrics for quick ablations."""

from __future__ import annotations

import torch
from torch import Tensor


def token_accuracy(logits: Tensor, target: Tensor, pad_idx: int) -> float:
    pred = logits.argmax(dim=-1)
    valid = target != pad_idx
    correct = (pred == target) & valid
    denom = torch.clamp(valid.sum(), min=1)
    return (correct.sum().float() / denom.float()).item()
