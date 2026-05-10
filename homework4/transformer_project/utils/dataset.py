"""Dataset helpers for Multi30k translation experiments."""

from __future__ import annotations
import os
import torch
from dataclasses import dataclass
from torch import Tensor
from torch.utils.data import Dataset

@dataclass
class Pair:
    src: str
    tgt: str

class TranslationDataset(Dataset):
    def __init__(self, pairs: list[Pair], src_tokenizer, tgt_tokenizer, max_len: int = 64) -> None:
        self.pairs = pairs
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def _pad(self, ids: list[int], pad_idx: int) -> list[int]:
        # 截断到 max_len - 2 (预留给 BOS 和 EOS) 或直接截断
        ids = ids[: self.max_len]
        return ids + [pad_idx] * (self.max_len - len(ids))

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        pair = self.pairs[idx]
        
        # 从 tokenizer 获取特殊 token 的 ID
        # 注意：这里假设你的 tokenizer 属性名是 bos_token_id 等，请根据实际 tokenizer.py 修改
        bos_id = self.src_tokenizer.stoi[self.src_tokenizer.bos_token]
        eos_id = self.src_tokenizer.stoi[self.src_tokenizer.eos_token]
        pad_id = self.src_tokenizer.stoi[self.src_tokenizer.pad_token]

        # 编码并添加 BOS/EOS
        src_ids = [bos_id] + self.src_tokenizer.encode(pair.src) + [eos_id]
        tgt_ids = [bos_id] + self.tgt_tokenizer.encode(pair.tgt) + [eos_id]
        
        # 填充
        src_padded = self._pad(src_ids, pad_id)
        tgt_padded = self._pad(tgt_ids, pad_id)
        
        return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)

def load_multi30k_pairs(split: str, limit: int | None = None) -> list[Pair]:
    """从本地 data/ 目录加载 Multi30k 数据"""
    file_prefix = split if split != 'valid' else 'val'
    
    # 路径拼接
    src_path = os.path.join("data", f"{file_prefix}.de")
    tgt_path = os.path.join("data", f"{file_prefix}.en")
    
    pairs: list[Pair] = []
    
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        raise FileNotFoundError(f"找不到数据文件: {src_path} 或 {tgt_path}。请确保文件在 data 文件夹下。")

    with open(src_path, 'r', encoding='utf-8') as f_de, \
         open(tgt_path, 'r', encoding='utf-8') as f_en:
        
        for i, (de_line, en_line) in enumerate(zip(f_de, f_en)):
            src = de_line.strip().lower()
            tgt = en_line.strip().lower()
            
            if src and tgt:
                pairs.append(Pair(src=src, tgt=tgt))
            
            if limit is not None and len(pairs) >= limit:
                break
                
    print(f"成功从本地加载 {len(pairs)} 条 {split} 数据。")
    return pairs
