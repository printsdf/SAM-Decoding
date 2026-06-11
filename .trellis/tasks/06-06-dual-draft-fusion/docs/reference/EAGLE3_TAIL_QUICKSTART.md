# EAGLE3 Tail 快速入门

## TL;DR

EAGLE3 Tail 已集成到 SAM-Decoding 中。添加 `--eagle3_tail_path` 即可启用；复现官方 tail 结果时确认 `total_token=60, depth=7, top_k=10`。

## 1 分钟快速测试

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，设置 MODEL_PATH, TREE_MODEL_PATH, EAGLE3_TAIL_PATH

# 2. 运行简单测试
bash scripts/test_tail_simple.sh
```

## 核心概念

**问题**: EAGLE3 原版只能预测 32K draft vocabulary，无法覆盖完整的 128K+ 目标词汇表。

**解决方案**: Tail sidecar 通过低秩分解预测 V_miss（缺失词汇），与原始 lm_head 输出合并。

```
┌─────────────────────────────────────┐
│  EAGLE3 Hidden States (4096-d)     │
└──────────┬──────────────────────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌─────────┐
│lm_head  │  │  Tail   │
│(32K)    │  │(96K)    │
└────┬────┘  └────┬────┘
     │            │
     └─────┬──────┘
           ▼
    ┌──────────────┐
    │ Full Logits  │
    │  (128K)      │
    └──────────────┘
```

## 使用方式

### 方式 1: 通过环境变量（推荐）

**编辑 `.env`:**
```bash
EAGLE3_TAIL_PATH=./tail_epoch_10.pt
EAGLE3_TAIL_TYPE=auto
```

**运行:**
```bash
bash scripts/test_samd_eagle3.sh
```

### 方式 2: 命令行参数

```bash
python evaluation/inference_samd.py \
    --model-path /path/to/Meta-Llama-3.1-8B-Instruct \
    --model-type llama3 \
    --model-id test \
    --bench-name medqa \
    --tree_method eagle3 \
    --tree_model_path /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
    --eagle3_tail_path ./tail_epoch_10.pt \
    --eagle3_tail_type auto \
    --eagle3_total_token 60 \
    --eagle3_depth 7 \
    --eagle3_top_k 10
```

### 方式 3: Python API

```python
from samd import SamdConfig, SamdModel, DraftModel

samd_config = SamdConfig(
    tree_method="eagle3",
    tree_model_path="/path/to/EAGLE3-LLaMA3.1-Instruct-8B",
    eagle3_tail_path="./tail_epoch_10.pt",  # 添加这一行
    eagle3_tail_type="auto",                # 添加这一行
    eagle3_total_token=60,
    eagle3_depth=7,
    eagle3_top_k=10,
)
# ... 其余代码不变
```

## 验证 Checkpoint

使用内置工具检查 tail checkpoint：

```bash
python scripts/inspect_tail_checkpoint.py ./tail_epoch_10.pt
```

**期望输出:**
```
Checkpoint keys and shapes:
  down.weight          [512, 4096]               dtype=torch.float32
  up.weight            [96256, 512]              dtype=torch.float32
  ...

Tail type detection:
  Type: PlainTail
  hidden_size: 4096
  rank: 512
  n_vmiss: 96256
  ✓ Checkpoint matches LLaMA-3.1-8B vocab structure
```

**关键检查点:**
- ✓ `n_vmiss = 96256` (对于 LLaMA-3.1-8B: 128256 - 32000)
- ✓ `hidden_size = 4096` (匹配你的模型)
- ✓ `acc > 0.5` (训练准确率)

## 性能对比

运行自动化对比脚本：

```bash
bash scripts/compare_baseline_tail.sh
```

**预期结果 (MedQA):**
```
Accept Length Improvement:
  Baseline: 3.099
  Tail:     3.638
  Gain:     +17.4%

Expected improvement: +17.4% ~ +17.5%
✓ Tail is providing significant improvement!
```

## 与 SAM Fusion 结合

Tail 可以与 SAM tree fusion 策略无缝结合：

```bash
python evaluation/inference_samd.py \
    --tree_method eagle3 \
    --tree_fusion sam_sequence_graft \
    --eagle3_tail_path ./tail_epoch_10.pt \
    --eagle3_total_token 60 \
    --eagle3_depth 7 \
    --eagle3_top_k 10 \
    --sam_tree_max_nodes 16 \
    --sam_tree_top_k 4 \
    --sam_tree_alpha 4.0 \
    ...
```

## 常见问题

### Q1: 如何知道 tail 是否生效？

**检查日志输出:**
```
attached EAGLE3 plain tail sidecar: vocab_size=128256, original_draft_vocab_size=32000, vmiss=96256
```

如果看到这条日志，说明 tail 已成功加载。

### Q2: 加速比没有达到预期怎么办？

**调试步骤:**

1. **检查 checkpoint 质量:**
   ```bash
   python scripts/inspect_tail_checkpoint.py ./tail_epoch_10.pt
   # 确保 acc > 0.5
   ```

2. **运行 baseline 对比:**
   ```bash
   bash scripts/compare_baseline_tail.sh
   # 查看具体差异
   ```

3. **检查模型路径匹配:**
   - Tail 必须用与你当前使用相同的 MODEL_PATH 和 TREE_MODEL_PATH 训练
   - 词汇表大小必须完全一致

### Q3: 支持哪些模型？

目前支持：
- ✓ LLaMA-3 / LLaMA-3.1 系列
- ✓ 任何使用 LlamaForCausalLM 架构的模型

**要求:**
- EAGLE3 tree model 已训练好
- Tail checkpoint 与目标模型词汇表匹配

### Q4: PlainTail 和 TuckerTail 有什么区别？

| 特性 | PlainTail | TuckerTail |
|------|-----------|------------|
| 参数量 | 较多 | 较少 |
| 精度 | 略高 | 略低 |
| 性能 | +17.4% | +17.5% |
| 推荐 | 有足够显存时 | 显存受限时 |

使用 `--eagle3_tail_type auto` 自动检测。

## 代码位置

如果你想了解实现细节：

- **核心实现**: `samd/tree_model/eagle3/tail_sidecar.py`
  - `PlainTail`: 低秩分解 (line 10-17)
  - `TuckerTail`: Tucker 分解 (line 20-46)
  - `CombinedHead`: 合并输出 (line 49-97)
  - `attach_tail_sidecar`: 附加函数 (line 152-189)

- **集成点**: `samd/tree_model/eagle3/eagle3.py:45-52`
  ```python
  if config.eagle3_tail_path is not None:
      attach_tail_sidecar(...)
  ```

- **配置**: `samd/samd_config.py:40-41`
  ```python
  eagle3_tail_path: Optional[str] = field(default=None)
  eagle3_tail_type: Literal["auto", "plain", "tucker"] = field(default="auto")
  ```

## 下一步

- 阅读完整文档: `EAGLE3_TAIL_USAGE.md`
- 🔬 运行性能对比: `bash scripts/compare_baseline_tail.sh`
- 🧪 尝试 SAM fusion: 添加 `--tree_fusion sam_sequence_graft`
- 🐛 遇到问题: 运行 `python scripts/diagnose_tail.py`

## 贡献

基于 EAGLE3 官方实现，集成到 SAM-Decoding 框架。

**参考文献:**
- EAGLE3 论文: [链接]
- 官方代码: https://github.com/SafeAILab/EAGLE
