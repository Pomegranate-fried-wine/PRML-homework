# transformer_project (Multi30k)

在 `main/homework4` 下实现本地小型 Transformer，使用 **Multi30k(de->en)** 进行训练、评估、消融和可视化。

---
## 1. 代码结构与主要功能

### `model/`
- `attention.py`：多头注意力 `MultiHeadAttention`，返回 `output + attention map`。
- `encoder.py`：`EncoderLayer/Encoder`，包含 self-attention + MLP(FFN) + LayerNorm + 残差（可开关）。
- `decoder.py`：`DecoderLayer/Decoder`，包含 masked self-attn + cross-attn + FFN + 残差（可开关）。
- `transformer.py`：总模型拼接（Embedding + PositionalEncoding + Encoder + Decoder + Linear）。
  - 位置编码支持：`sinusoidal / learned / none`。

### `utils/`
- `tokenizer.py`：`BasicTokenizer`（空格切分 + BOS/EOS/PAD/UNK）。
- `dataset.py`：Multi30k 读取与 `TranslationDataset` 封装。
- `metrics.py`：token-level accuracy。

### 顶层脚本
- `train.py`：训练与验证，保存 best checkpoint，输出并记录 loss/acc 曲线数据。
- `eval.py`：greedy decode + BLEU 评估，导出 JSON（含样例预测）。
- `visualize_attention.py`：图表可视化（注意力热力图 / 训练曲线 / 对照组BLEU柱状图）。
- `experiments/no_pe.py`：位置编码对照实验。
- `experiments/ablation_residual.py`：残差消融实验。

---
## 2. 项目输出信息包括什么

### 训练阶段输出（`train.py`）
- 终端每 epoch 输出：
  - `train_loss`
  - `train_acc`
  - `val_loss`
  - `val_acc`
- 文件输出：
  - `checkpoints/*.pt`（最佳模型）
  - `logs/train_metrics.csv`（每 epoch 指标，用于作图）

### 评估阶段输出（`eval.py`）
- 终端输出：`BLEU=...`。
- JSON输出（如 `logs/eval_metrics.json`）：
  - `bleu`
  - `num_samples`
  - `pe_mode`
  - `use_residual`
  - `examples`（预测/参考翻译样例）

### 可视化输出（`visualize_attention.py`）
- 注意力热力图：`attention_map.png`
- 训练曲线图（loss/acc）：如 `figures/train_curves.png`
- 对照实验BLEU柱状图：如 `figures/ablation_bleu.png`

---
## 3. 模型效果指标说明

### 主指标
- **BLEU**（句子级翻译质量的语料级统计指标，核心报告指标）。

### 辅助指标
- **train/val loss**：收敛速度与稳定性。
- **token accuracy**：逐 token 预测正确率（训练过程观察用）。

> 解释：BLEU更贴近最终翻译质量；loss/acc更适合诊断训练是否稳定。

---
## 4. 对照组（ablation）输出信息包括什么

### 位置编码对照（`experiments/no_pe.py`）
- 组别：
  1. `sinusoidal`
  2. `learned`
  3. `none`
- 每组输出：
  - 各 epoch 的 train/val loss + acc（CSV）
  - BLEU（JSON + 终端）
  - 可绘制跨组 BLEU 柱状图（bar）

### 残差对照（`experiments/ablation_residual.py`）
- 组别：
  1. residual ON
  2. residual OFF
- 每组输出：
  - 训练曲线（loss/acc）
  - BLEU
  - 样例预测对比（JSON中的 examples）

---
## 5. 快速运行命令

### 安装依赖
```bash
pip install torch torchtext nltk matplotlib
```

### 基线训练
```bash
python train.py --epochs 10 --train_limit 12000 --valid_limit 1000 --save checkpoints/base.pt --log_csv logs/base_train.csv
```

### 基线评估
```bash
python eval.py --ckpt checkpoints/base.pt --test_limit 1000 --save_json logs/base_eval.json
```

### 图表1：注意力热力图
```bash
python visualize_attention.py --mode attention --ckpt checkpoints/base.pt --index 0 --out figures/attention_map.png
```

### 图表2：训练曲线
```bash
python visualize_attention.py --mode curves --csv logs/base_train.csv --out figures/base_train_curves.png
```

### 图表3：对照组BLEU柱状图
```bash
python visualize_attention.py --mode bar --eval_jsons logs/base_eval.json logs/pe_learned_eval.json logs/pe_none_eval.json --out figures/ablation_bleu.png
```

### 运行对照实验
```bash
python experiments/no_pe.py
python experiments/ablation_residual.py
```
