# 《Attention Is All You Need》(2017) 复现与模块必要性实验报告（Homework 4）

## 1. 作业要求对应关系

本报告严格对应作业两项要求：

1. **阅读并复现经典论文《Attention Is All You Need》**：
   在 Multi30k de→en 翻译任务上，实现并训练 Encoder-Decoder Transformer（多头注意力、FFN、残差、LayerNorm、位置编码、masked decoding）。
2. **通过实验理解模块必要性**：
   本项目实际完成了两类消融：
   - 作业2.1 位置编码：`sinusoidal / learned absolute / none`；
   - 作业2.3 ResNet 式残差连接：`residual on / residual off`。

---

## 2. 项目代码结构与各模块作用梳理

### 2.1 数据与预处理

- `utils/tokenizer.py`：实现 `BasicTokenizer`，负责分词、词表构建及 `BOS/EOS/PAD/UNK` 处理。
- `utils/dataset.py`：读取本地 Multi30k 数据，构建 `TranslationDataset`，并在 batch 中进行 padding 与张量化。
- `data/*.de, *.en`：训练/验证/测试平行语料文件。

### 2.2 模型实现（Transformer 复现）

- `model/attention.py`：实现 Scaled Dot-Product + Multi-Head Attention，并返回 attention map（用于解释性可视化）。
- `model/encoder.py`：EncoderLayer/Encoder 堆叠，包含 self-attn + FFN + Add&Norm（残差可开关）。
- `model/decoder.py`：DecoderLayer/Decoder 堆叠，包含 masked self-attn + cross-attn + FFN + Add&Norm（残差可开关）。
- `model/transformer.py`：总装模块（Embedding + PositionalEncoding + Encoder + Decoder + Linear），位置编码支持 `sinusoidal / learned / none`。

### 2.3 训练、评估、实验与可视化脚本

- `train.py`：训练与验证主入口，输出并记录 `train/val loss` 与 `train/val acc`，保存最佳 checkpoint。
- `eval.py`：greedy decode + BLEU 评估，输出 JSON 指标和样例。
- `experiments/no_pe.py`：自动跑位置编码对照实验。
- `experiments/ablation_residual.py`：自动跑残差开关消融实验。
- `visualize_attention.py`：绘制注意力热力图、训练曲线图、BLEU 柱状图。

---

## 3. 实验设置

- 任务：Multi30k 德译英（de→en）。
- 数据量：train 12,000；valid 1,000；test 1,000。
- 统一训练配置：10 epochs，batch size=64，Adam，lr=2e-4。
- 指标：
  - 主指标：**BLEU**（翻译质量）；
  - 辅助指标：loss、token accuracy（观察收敛行为）。

---

## 4. 实验一：位置编码必要性（sinusoidal vs learned vs none）

### 4.1 实验动机

论文指出 attention 本身不编码顺序，因此必须注入位置信息。本实验比较固定正弦位置编码、可学习绝对位置编码，以及完全去掉位置编码。

### 4.2 实验结果（你提供日志整理）

| PE 模式 | train_loss@10 | train_acc@10 | val_loss@10 | val_acc@10 | test BLEU |
|---|---:|---:|---:|---:|---:|
| sinusoidal | 1.8012 | 0.6512 | 2.1028 | 0.6263 | 0.2452 |
| learned | 1.8470 | 0.6455 | 2.1514 | 0.6186 | **0.2532** |
| none | 2.0245 | 0.6023 | 2.4052 | 0.5259 | 0.0001 |

### 4.3 现象分析

1. **无位置编码导致“会选词，不会排词序”**：
   `none` 的训练 acc 仍上升到 0.60 左右，但 BLEU≈0，说明模型可记住词频与局部对应，却无法形成正确句法顺序和短语结构。

2. **sinusoidal 与 learned 都能有效工作**：
   两者在 loss/acc 曲线上接近，说明二者都成功把“顺序信息”注入注意力计算。

3. **本次数据规模下 learned 略优**：
   `learned` BLEU=0.2532 高于 `sinusoidal` 的 0.2452，可能因为可学习参数更贴合当前语料分布与句长统计。

### 4.4 结论

**位置编码的核心作用是为 self-attention 提供序列顺序约束，使模型能区分 token 的相对/绝对位置。** 缺失位置编码会使 Transformer 在翻译任务上退化为“弱语序模型”。

---

## 5. 实验二：ResNet 式残差连接必要性（on vs off）

### 5.1 实验动机

论文中的每个子层都采用 Add&Norm（含 residual）。残差通路可稳定深层训练并改善梯度传播。为验证其必要性，本实验对比 residual on/off。

### 5.2 残差消融结果

| 残差配置 | epoch1 train_loss / acc | epoch10 train_loss / acc | epoch10 val_loss / acc | test BLEU |
|---|---|---|---|---:|
| residual ON | 4.5351 / 0.3380 | 1.8064 / 0.6508 | 2.1168 / 0.6213 | 0.2308 |
| residual OFF | 5.4356 / 0.1350 | 4.6411 / 0.1427 | 4.6679 / 0.1409 | 0.0000 |

### 5.3 现象分析

1. **收敛速度显著下降**：
   residual off 从第1轮到第10轮 loss 仅小幅下降（5.4356→4.6411），表现为严重欠学习。

2. **梯度传播与表示学习受阻**：
   关闭残差后 train/val acc 长期停在 0.14 左右，接近低水平随机猜测区间，说明深层优化路径被明显破坏。

3. **最终翻译能力崩溃**：
   BLEU=0，并触发 4-gram overlap 警告，表明输出句子几乎不具备可用翻译质量。

### 5.4 结论（对应作业问题2）

**不采用 ResNet（残差）结构会导致深层 Transformer 难以训练，出现收敛慢、精度低、BLEU 归零等问题。** 残差的作用可概括为：稳定训练、改善梯度传播、提升收敛速度与最终性能。

---

## 6. 可视化结果说明

### 6.1 注意力热力图

![Cross-Attention Heatmap](figures/attention_map.png)

热力图显示目标词生成时对源词存在可解释的对齐趋势，说明 cross-attention 学到了跨语种词对齐关系。

### 6.2 残差开关训练曲线

![Residual ON Curves](figures/residual_on_curves.png)

![Residual OFF Curves](figures/residual_off_curves.png)

曲线与表格结论一致：残差开启时稳定下降并提升准确率，残差关闭时训练停滞。

### 6.3 BLEU 柱状图

![Ablation BLEU](figures/ablation_bleu.png)

柱状图直观支持两点：
- 有位置编码显著优于无位置编码；
- 有残差显著优于无残差。

---

## 7. 实验总结与反思

1. **对论文机制的理解从“结构层面”推进到“因果层面”**：
   复现后再做消融，能够明确看到“位置编码/残差”并非经验技巧，而是 Transformer 可用性的必要条件。

2. **指标解读的反思**：
   仅看 token accuracy 会误判模型效果（如 no-PE 有一定 acc 但 BLEU 几乎为 0）。翻译任务必须以 BLEU 或序列级指标作为主判断依据。

3. **实验设计上的收获**：
   对照实验必须控制变量（数据、epoch、优化器、batch 一致），否则难以把性能差异归因到单一模块。

4. **局限性与改进方向**：
   - 当前仅使用 10 epochs 和较小数据子集，结论在更大规模语料上仍可继续验证；
   - 评估采用 greedy decoding，可进一步加入 beam search；
   - 可增加相对位置编码（如 RoPE/ALiBi 风格）或 pre-norm/post-norm 对照，深化模块理解。

---

## 8. 最终结论

- 已完成《Attention Is All You Need》核心架构复现。
- 已通过两组消融实验验证模块必要性：
  1. **位置编码是语序建模前提**；
  2. **残差连接是深层训练稳定性的关键**。
