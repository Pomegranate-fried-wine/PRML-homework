# Transformer Reproduction Report (Homework4)

## Goal
- 在本地（RTX 4060）完成小型 Transformer 从零实现与经典论文核心结构复现。

## Suggested Setup
- Python 3.10+
- PyTorch 2.x
- CUDA 12.x

## Baseline Config (4060-Friendly)
- d_model: 256
- n_heads: 4
- n_layers: 3
- d_ff: 1024
- batch_size: 32
- seq_len: 24

## Dataset Plan
- 第一阶段使用 toy 平行语料保证链路可跑通。
- 第二阶段可替换为 Multi30k / IWSLT14 小规模子集。

## Experiments
1. No positional encoding (`experiments/no_pe.py`)
2. Remove residual connections (`experiments/ablation_residual.py`)
