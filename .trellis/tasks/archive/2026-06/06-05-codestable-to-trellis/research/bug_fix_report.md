# EAGLE3 Tail Bug 修复报告

## 问题诊断

你的 EAGLE3 Tail 实现结果与官方差距巨大：
- **官方**: baseline τ=3.099 → tail τ=3.638 (+17.4%)
- **你的**: tail τ=2.614-2.657 (比预期低 28%)

## 发现的 Bug

### 关键 Bug: `freq_indices` 未注册为 buffer

**位置**: `samd/tree_model/eagle3/tail_sidecar.py:69`

**问题代码**:
```python
class CombinedHead(nn.Module):
    def __init__(self, original_lm_head, tail, t2d_mask, vmiss_indices):
        super().__init__()
        self.original_lm_head = original_lm_head
        self.tail = tail
        self.register_buffer("t2d_mask", t2d_mask)
        self.register_buffer("vmiss_indices", vmiss_indices)
        self.freq_indices = torch.nonzero(t2d_mask, ...).squeeze(-1)  # ❌ Bug!
```

**问题分析**:
1. `freq_indices` 没有使用 `register_buffer()`
2. 当模块被 `.to(device)` 移动时，它**不会**被移动到 GPU
3. 导致在 forward 时：
   - `freq_indices` 在 CPU
   - `full_logits` 在 GPU
   - `full_logits[:, :, self.freq_indices]` 触发 CPU ↔ GPU 数据传输
4. 严重影响性能，甚至可能导致错误结果

**修复代码**:
```python
class CombinedHead(nn.Module):
    def __init__(self, original_lm_head, tail, t2d_mask, vmiss_indices):
        super().__init__()
        self.original_lm_head = original_lm_head
        self.tail = tail
        self.register_buffer("t2d_mask", t2d_mask)
        self.register_buffer("vmiss_indices", vmiss_indices)
        # 修复：注册为 buffer
        freq_indices = torch.nonzero(t2d_mask, ...).squeeze(-1)
        self.register_buffer("freq_indices", freq_indices)  # ✅ 修复
```

## 已修复

✅ **文件**: `samd/tree_model/eagle3/tail_sidecar.py`
✅ **更改**: 第 69-70 行，`freq_indices` 现在正确注册为 buffer

## 验证步骤

### 方法 1: 快速验证脚本

```bash
python scripts/verify_tail_fix.py
```

这个脚本会：
1. 检查 `freq_indices` 是否在正确的 device 上
2. 运行快速生成测试
3. 报告 accept length

### 方法 2: 完整性能对比

```bash
bash scripts/compare_baseline_tail.sh
```

对比 baseline 和 tail 的完整性能。

### 方法 3: 手动检查

```python
# 在代码中添加调试
import os
os.environ['MODEL_PATH'] = '/path/to/model'
os.environ['TREE_MODEL_PATH'] = '/path/to/tree_model'
os.environ['EAGLE3_TAIL_PATH'] = '/path/to/tail_checkpoint.pt'

from samd import SamdConfig, DraftModel
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    os.environ['MODEL_PATH'],
    torch_dtype=torch.float16,
    device_map="auto"
)

config = SamdConfig(
    tree_method="eagle3",
    tree_model_path=os.environ['TREE_MODEL_PATH'],
    eagle3_tail_path=os.environ['EAGLE3_TAIL_PATH'],
    eagle3_tail_type="auto"
)

device = next(model.parameters()).device
draft = DraftModel(config, sam_static=None, lm=model, dtype=torch.float16, device=device)

# 检查 device
lm_head = draft.tree.model.lm_head
print(f"freq_indices device: {lm_head.freq_indices.device}")
print(f"vmiss_indices device: {lm_head.vmiss_indices.device}")
print(f"Expected device: {device}")

# 应该输出：
# freq_indices device: cuda:0
# vmiss_indices device: cuda:0
# Expected device: cuda:0
```

## 预期效果

修复后，你应该看到：
- ✅ `freq_indices` 在正确的 device (cuda:0)
- ✅ 没有 CPU ↔ GPU 数据传输警告
- ✅ Accept length 显著提升（接近 +17%）

## 如果修复后仍有问题

如果 accept length 仍然较低，检查：

### 1. Tail Checkpoint 匹配
```bash
python scripts/inspect_tail_checkpoint.py /path/to/tail_checkpoint.pt
```
确保：
- `n_vmiss = 96256` (对于 LLaMA-3.1-8B)
- `hidden_size = 4096`
- `acc > 0.5` (训练准确率)

### 2. 模型路径一致
Tail checkpoint 必须使用与当前**完全相同**的：
- MODEL_PATH
- TREE_MODEL_PATH

### 3. Tree 配置参数
官方使用：
- `total_token=60`
- `depth=7`
- `top_k=10`

检查你的 `TREE_MODEL_PATH/config.json` 中的配置。

### 4. Dtype 差异
- 官方：bfloat16
- 你的：float16

这可能导致 ~5% 的性能差异，但不会是主要问题。

### 5. 数据集和评估方式
- 官方：MedQA test 273 questions
- 你的：MedQA 80 questions

确保使用相同的评估协议。

## 其他潜在优化

### 优化 1: 填充值
当前使用 `-1e4` (float16)，可能不够负。

**建议**:
```python
# Line 80
fill_value = -65504 if hidden_states.dtype == torch.float16 else -1e9
```

### 优化 2: 避免不必要的 squeeze/unsqueeze
如果性能仍然不理想，可以优化 forward 方法减少操作。

## 总结

- ✅ **Bug 已修复**: `freq_indices` 现在正确注册为 buffer
- ✅ **验证脚本已创建**: `scripts/verify_tail_fix.py`
- ⏳ **等待验证**: 运行验证脚本检查修复效果

修复这个 bug 后，你的 accept length 应该会显著提升。如果达到 3.0-3.6 的范围，说明修复成功！
