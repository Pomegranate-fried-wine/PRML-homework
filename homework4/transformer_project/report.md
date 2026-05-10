# Transformer Reproduction Report (Homework4)

## 数据集设置
- 使用 `torchtext.datasets.Multi30k`，翻译方向：`de -> en`。
- 默认训练子集：12000（train）/ 1000（valid）/ 1000（test），适配 RTX 4060 显存。

## 问题1：位置编码的核心作用
### 对比设置
1. **Sinusoidal**（原论文默认）
2. **Learned absolute PE**
3. **None**（不使用位置编码）

### 讨论要点
- 注意力本身对 token 顺序不敏感，位置编码为模型注入序列顺序信息。
- Sinusoidal 编码对长度外推较友好；learned 编码常在固定长度任务中表现更灵活。
- 无位置编码时，语序建模显著受损，BLEU 通常明显下降。

## 问题2：ResNet残差连接消融
### 对比设置
1. Residual ON
2. Residual OFF

### 讨论要点
- 残差连接提升梯度传播稳定性，尤其在多层编码器/解码器时更明显。
- 去掉残差通常会导致收敛变慢、训练波动增大和最终指标下降。

## 问题3：评估与可视化补齐
- `eval.py`：实现 greedy decoding + BLEU（`nltk`）评估。
- `visualize_attention.py`：导出 cross-attention 热力图（head 0）。
- `experiments/no_pe.py` 与 `experiments/ablation_residual.py`：自动训练并评估对比。
