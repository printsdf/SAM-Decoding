# EAGLE3 Tail 使用指南

## 功能概述

EAGLE3 Tail 是一个扩展模块，用于将 EAGLE3 的预测能力从原始的 32K draft vocabulary 扩展到完整的目标模型词汇表（如 LLaMA-3.1-8B 的 128256 tokens）。

### 架构说明

- **PlainTail**: 使用低秩分解 `hidden_states -> rank -> V_miss`
- **TuckerTail**: 使用 Tucker 分解，更加参数高效
- **CombinedHead**: 将原始 lm_head（32K）和 tail（V_miss）的输出合并为完整词汇表

### 关键参数

- `V_miss`: 缺失词汇数量 = target_vocab_size - draft_vocab_size
  - 例如 LLaMA-3.1-8B: 128256 - 32000 = 96256

## 使用方法

### 1. 准备 Tail Checkpoint

确保你有训练好的 tail checkpoint（例如 `tail_epoch_10.pt`），checkpoint 应包含：

**PlainTail 格式:**
```python
{
    "down.weight": [rank, hidden_size],
    "up.weight": [n_vmiss, rank],
    "epoch": 10,
    "loss": 0.xxxx,
    "acc": 0.xxxx
}
```

**TuckerTail 格式:**
```python
{
    "core.weight": [r1*r2, hidden_size],
    "U1": [v1, r1],
    "U2": [v2, r2],
    "epoch": 10,
    "loss": 0.xxxx,
    "acc": 0.xxxx
}
```

### 2. 配置环境变量

编辑 `.env` 文件：

```bash
# 基础模型路径
MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct
TREE_MODEL_PATH=/path/to/EAGLE3-LLaMA3.1-Instruct-8B

# Tail sidecar 配置
EAGLE3_TAIL_PATH=./tail_epoch_10.pt
EAGLE3_TAIL_TYPE=auto  # auto | plain | tucker
```

### 3. 命令行使用

#### 测试脚本
```bash
# 使用环境变量配置
bash scripts/test_samd_eagle3.sh

# 或直接指定参数
python -m tests.test_samd \
    --model_path /path/to/Meta-Llama-3.1-8B-Instruct \
    --tree_method eagle3 \
    --tree_model_path /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
    --eagle3_tail_path ./tail_epoch_10.pt \
    --eagle3_tail_type auto \
    --eagle3_total_token 60 \
    --eagle3_depth 7 \
    --eagle3_top_k 10 \
    --dtype float16
```

#### 评估脚本
```bash
python evaluation/inference_samd.py \
    --model-path /path/to/Meta-Llama-3.1-8B-Instruct \
    --model-type llama3 \
    --model-id llama3-eagle3-tail \
    --bench-name medqa \
    --tree_method eagle3 \
    --tree_model_path /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
    --eagle3_tail_path ./tail_epoch_10.pt \
    --eagle3_tail_type auto \
    --eagle3_total_token 60 \
    --eagle3_depth 7 \
    --eagle3_top_k 10 \
    --dtype float16 \
    --max-new-tokens 512
```

#### CLI 交互模式
```bash
python -m samd.inference.cli \
    --model /path/to/Meta-Llama-3.1-8B-Instruct \
    --tree_method eagle3 \
    --tree_model_path /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
    --eagle3_tail_path ./tail_epoch_10.pt \
    --eagle3_tail_type auto \
    --eagle3_total_token 60 \
    --eagle3_depth 7 \
    --eagle3_top_k 10
```

### 4. Python API 使用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from samd import SamdConfig, SamdModel, DraftModel, SamdGenerationConfig

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/path/to/Meta-Llama-3.1-8B-Instruct")

# 配置 SAMD + EAGLE3 + Tail
samd_config = SamdConfig(
    tree_method="eagle3",
    tree_model_path="/path/to/EAGLE3-LLaMA3.1-Instruct-8B",
    eagle3_tail_path="./tail_epoch_10.pt",
    eagle3_tail_type="auto",  # 自动检测 plain 或 tucker
    eagle3_total_token=60,
    eagle3_depth=7,
    eagle3_top_k=10,
)

# 创建 draft 和 SAMD 模型
draft = DraftModel(
    samd_config,
    sam_static=None,
    lm=model,
    dtype=torch.float16,
    device="cuda"
)

samd_model = SamdModel(
    samd_config,
    model,
    draft,
    tokenizer.eos_token_id,
    torch.float16,
    "cuda"
)

# 生成
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = samd_model.generate(
    **inputs,
    generation_config=SamdGenerationConfig(
        max_new_tokens=512,
        max_cache_len=4096,
        greedy=True,
        temperature=0.0
    )
)

response = tokenizer.decode(outputs.output_ids[0])
print(response)
print(f"Decode steps: {outputs.decode_steps}")
print(f"Accept length per step: {outputs.accepet_length_per_step}")
```

## 工作原理

### 1. 加载阶段 (`eagle3.py:45-54`)

```python
if config.eagle3_tail_path is not None:
    attach_tail_sidecar(
        self.model,
        tail_path=config.eagle3_tail_path,
        tail_type=config.eagle3_tail_type,
        dtype=dtype,
        device=device,
    )
```

### 2. Tail 附加过程 (`tail_sidecar.py:152-189`)

1. 从 checkpoint 推断 tail 类型（plain 或 tucker）
2. 构建 tail 模块并加载权重
3. 创建 `CombinedHead`，包装原始 `lm_head` 和 `tail`
4. 更新 EAGLE3 的配置：
   - `draft_vocab_size`: 32000 → 128256
   - `d2t`: identity mapping `[0, 1, 2, ..., vocab_size-1]`
   - `t2d`: 全部标记为 True（所有 token 都可用）

### 3. 推理阶段 (`tail_sidecar.py:71-92`)

```python
# CombinedHead.forward
main_logits = self.original_lm_head(hidden_states)  # [B, S, 32000]
tail_logits = self.tail(hidden_states)              # [B, S, 96256]

# 合并到完整词汇表
full_logits = torch.full((B, S, 128256), -1e9, ...)
full_logits[:, :, freq_indices] = main_logits       # 填充高频 32K
full_logits[:, :, vmiss_indices] = tail_logits      # 填充低频 96K
```

## 诊断工具

### 检查 Checkpoint 结构

```bash
python scripts/inspect_tail_checkpoint.py ./tail_epoch_10.pt
```

输出示例：
```
Checkpoint keys and shapes:
  down.weight          [512, 4096]               dtype=torch.float32
  up.weight            [96256, 512]              dtype=torch.float32
  epoch                10
  loss                 1.2345

Tail type detection:
  Type: PlainTail
  hidden_size: 4096
  rank: 512
  n_vmiss: 96256
  Expected for LLaMA-3.1-8B: n_vmiss=96256 (128256-32000)
  ✓ Checkpoint matches LLaMA-3.1-8B vocab structure
```

### 诊断加速比问题

```bash
python scripts/diagnose_tail.py
```

## 预期性能

根据官方 EAGLE3 论文，在 MedQA 数据集上：

| 配置 | Accept Length (τ) | 提升 |
|------|------------------|------|
| EAGLE3 baseline (32K) | 3.099 | - |
| + PlainTail | 3.638 | +17.4% |
| + TuckerTail | 3.640 | +17.5% |

**注意事项：**
- 官方使用 bfloat16，你可能使用 float16
- 官方 tree 参数：`total_token=60, depth=7, top_k=10`；SAMD 中对应 `--eagle3_total_token 60 --eagle3_depth 7 --eagle3_top_k 10`
- 如果你的加速比显著低于预期，检查：
  1. Tail checkpoint 是否匹配你的模型路径
  2. 树配置参数是否一致
  3. 先运行无 tail 的 baseline 对比

## 故障排查

### 问题 1: 加载失败

**错误:** `ValueError: plain tail V_miss mismatch`

**解决:**
- 检查 tail checkpoint 是否为正确的模型训练（使用 `inspect_tail_checkpoint.py`）
- 确保 `MODEL_PATH` 的词汇表大小与 checkpoint 匹配

### 问题 2: 加速比不理想

**可能原因:**
1. Tail checkpoint 训练质量不足（检查 `acc` 字段）
2. 树配置参数不匹配
3. dtype 差异（bfloat16 vs float16）

**调试步骤:**
```bash
# 1. 运行 baseline（无 tail）
python evaluation/inference_samd.py ... --eagle3_tail_path ""

# 2. 运行 tail 版本
python evaluation/inference_samd.py ... --eagle3_tail_path ./tail_epoch_10.pt

# 3. 对比 accept_length_per_step
```

### 问题 3: OOM

**解决:**
- 降低 `max_cache_len`: `--max_cache_len 4096`
- 使用 float16 而非 bfloat16
- 调整树配置减少节点数

## 与 SAM Fusion 结合使用

Tail 功能可以与 SAM tree fusion 策略结合：

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

支持的融合策略：
- `none`: 纯 EAGLE3 + Tail
- `sam_sequence_graft`: SAM 序列嫁接到 EAGLE3 树
- `sam_tree_union_prune`: SAM 树与 EAGLE3 树合并后剪枝
- `eagle_prefix_sam_expand`: EAGLE3 前缀 + SAM 扩展

## 参考文件

- 核心实现: `samd/tree_model/eagle3/tail_sidecar.py`
- 集成点: `samd/tree_model/eagle3/eagle3.py:45-52`
- 配置: `samd/samd_config.py:40-41, 62-65`
- 测试: `tests/test_samd.py`, `scripts/test_samd_eagle3.sh`
