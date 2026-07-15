# Phase 3.1 诊断方案：为什么 naive_logprob 比 pure Eagle3 慢？

## 问题陈述

Phase 3.1 结果（MT-Bench 完整）：
```
pure_eagle3:    55.917 TPS, 6.326 MAT (1.000x)
naive_logprob:  53.539 TPS, 6.310 MAT (0.957x) ❌ 慢 4.3%
```

**核心疑问**：
1. 为什么 naive_logprob 比 pure Eagle3 慢？
2. Eagle accept rate 8-9% 是否正确？
3. 融合路径本身是否有性能开销？

---

## 诊断实验设计

### Experiment 1: 融合路径开销测试

**目的**：测量融合路径本身的开销

**方案**：在同一环境运行三个配置

```bash
# Config A: Pure Eagle3 (baseline)
python evaluation/inference_samd.py \
  --bench_name mt_bench \
  --fusion_mode none \
  --samd_len_threshold 1000000 \
  --answer_file mt_bench_pure_eagle3.jsonl

# Config B: Fusion path but 100% skip SAM (测试融合路径开销)
python evaluation/inference_samd.py \
  --bench_name mt_bench \
  --fusion_mode naive \
  --samd_len_threshold 999999 \
  --answer_file mt_bench_fusion_100skip.jsonl

# Config C: Fusion path with normal threshold (当前配置)
python evaluation/inference_samd.py \
  --bench_name mt_bench \
  --fusion_mode naive \
  --samd_len_threshold 5 \
  --answer_file mt_bench_fusion_normal.jsonl
```

**预期结果**：

| Config | SAM Skip | Expected TPS | 结论 |
|--------|----------|--------------|------|
| A (pure) | N/A | 55.9 | Baseline |
| B (100% skip) | 100% | ~55.5 | 如果接近 A，说明融合路径开销小 |
| C (normal) | 76-98% | 53.5 | 当前结果 |

**判断标准**：
- 如果 `B ≈ A`（< 1% 差异）：融合路径开销可接受，问题在别处
- 如果 `B < A`（1-3% 慢）：融合路径有轻微开销
- 如果 `B << A`（> 3% 慢）：融合路径本身有问题，需要优化

---

### Experiment 2: Logprob 提取开销测试

**目的**：测量 logprob 提取的性能影响

**方案**：对比有/无 logprob 的 Eagle3

```python
# 修改代码添加开关
# samd/tree_model/eagle3/eagle3.py
def gen_draft(self, start_token: int, return_logprobs: bool = True):
    if return_logprobs:
        draft_tokens, buffers, logprobs = self.model.topK_genrate(...)
        return draft_tokens, buffers, logprobs
    else:
        # 不计算 logprobs
        draft_tokens, buffers = self.model.topK_genrate_no_logprob(...)
        return draft_tokens, buffers, None
```

**测试**：
```bash
# Without logprob (需要代码支持)
python evaluation/inference_samd.py \
  --fusion_mode naive \
  --disable_logprob \
  --answer_file mt_bench_no_logprob.jsonl

# With logprob (当前)
python evaluation/inference_samd.py \
  --fusion_mode naive \
  --answer_file mt_bench_with_logprob.jsonl
```

**预期**：logprob 提取开销应该 < 1%（因为只是 `torch.log(probs)`）

---

### Experiment 3: Eagle Accept Rate 验证

**目的**：验证 Eagle accept rate 统计是否正确

**方案**：添加详细的逐步日志

```python
# 在 samd/draft.py::record_naive_fusion_accept() 添加
def record_naive_fusion_accept(...):
    # 添加 debug 日志
    print(f"[ACCEPT] step={self.fusion_stats['steps']}")
    print(f"  selected_eagle={selected_eagle}, selected_sam={selected_sam}")
    print(f"  eagle_accepted={eagle_accepted}, sam_accepted={sam_accepted}")
    print(f"  eagle_rate={eagle_accepted/selected_eagle if selected_eagle > 0 else 0:.3f}")
    print(f"  sam_rate={sam_accepted/selected_sam if selected_sam > 0 else 0:.3f}")
    print(f"  accepted_indices={accepted_indices.tolist() if accepted_indices is not None else None}")
    print(f"  node_sources={node_sources[:10]}...")  # 前 10 个
```

**运行 3-5 样本**：
```bash
python evaluation/inference_samd.py \
  --bench_name mt_bench \
  --question_begin 0 \
  --question_end 5 \
  --fusion_mode naive \
  --answer_file mt_bench_debug_5.jsonl \
  2>&1 | tee debug_accept_rate.log

# 分析日志
grep "\[ACCEPT\]" debug_accept_rate.log | head -50
```

**检查**：
- `selected_eagle` 是否合理（应该约 40-50 个节点）
- `eagle_accepted` 是否合理（如果 accept rate=8%，应该约 3-4 个）
- `node_sources` 分布是否正确（eagle/sam/both）

---

### Experiment 4: 分数分布分析

**目的**：检查 logprob 归一化后的分数分布

**方案**：输出融合前后的分数统计

```python
# 在 samd/fusion/naive_fusion.py 添加
def fuse_eagle_sam_naive(...):
    # 解析前
    eagle_nodes = parse_eagle_tree(eagle_tree, eagle_logprobs)
    sam_nodes = parse_sam_sequence(sam_candidates)

    # 添加统计
    print(f"[SCORE] Eagle raw logprobs: min={min(n.score for n in eagle_nodes):.2f}, max={max(n.score for n in eagle_nodes):.2f}")
    print(f"[SCORE] SAM raw match_length: min={min(n.score for n in sam_nodes):.2f}, max={max(n.score for n in sam_nodes):.2f}")

    # 归一化后
    # ... normalize_scores(eagle_nodes, sam_nodes)

    print(f"[SCORE] Eagle normalized: min={min(n.norm_score for n in eagle_nodes):.2f}, max={max(n.norm_score for n in eagle_nodes):.2f}")
    print(f"[SCORE] SAM normalized: min={min(n.norm_score for n in sam_nodes):.2f}, max={max(n.norm_score for n in sam_nodes):.2f}")
```

**检查**：
- Logprob 范围是否合理（应该 [-20, 0]）
- 归一化后是否平衡（两边都在 [0, 1]）
- 是否有异常值（NaN, Inf）

---

## 实施优先级

### Priority 1（立即执行）：Experiment 1

**原因**：
- 最简单，不需要修改代码
- 直接回答核心问题：融合路径是否有开销
- 30 分钟即可完成

**执行**：
```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# Run Config B (100% skip)
python evaluation/inference_samd.py \
  --template llama3 \
  --model-type llama3 \
  --model-path "$MODEL_PATH" \
  --model-id mt_bench_fusion_100skip \
  --bench-name mt_bench \
  --fusion_mode naive \
  --samd_len_threshold 999999 \
  --answer-file evaluation/data/mt_bench/model_answer/fusion_100skip.jsonl \
  2>&1 | tee logs/fusion_100skip.log

# Compare with Config A (pure Eagle3)
# (已有数据：55.917 TPS)
```

### Priority 2（根据 P1 结果）：Experiment 3 或 4

**如果 P1 结果**：
- `Config B ≈ Config A`：问题不在融合路径 → 执行 Experiment 3（验证统计）
- `Config B < Config A`：融合路径有开销 → 需要 profile 找瓶颈

### Priority 3（可选）：Experiment 2

仅在其他实验无法定位问题时执行。

---

## 决策树

```
Experiment 1 结果
    ├─ Config B ≈ Config A (< 1% 差异)
    │   └─> 融合路径 OK
    │       └─> Experiment 3: 验证统计正确性
    │           ├─ 统计有误 → 修复统计
    │           └─ 统计正确 → Experiment 4: 检查分数分布
    │
    ├─ Config B 慢 1-3%
    │   └─> 融合路径有轻微开销
    │       └─> Profile 找瓶颈（logprob 提取、树解析、统计记录）
    │
    └─ Config B 慢 > 3%
        └─> 融合路径有严重问题
            └─> 详细 profile + 逐步优化
```

---

## 成功标准

**短期目标（本周）**：
- 找到 naive_logprob 比 pure Eagle3 慢的根本原因
- 修复问题或确认无法修复

**中期目标（下周）**：
- 如果能修复：继续 Phase 3.2（Payoff 校准）
- 如果无法修复：转向方案 A（enhanced graft）

---

## 时间估算

- Experiment 1: 30 分钟
- Experiment 3: 1 小时（修改代码 + 运行）
- Experiment 4: 1 小时
- **总计**: 2-3 小时可以完成完整诊断

---

## 下一步

请在远端执行 **Experiment 1**，把结果告诉我：

```bash
# Config B: 100% skip SAM
python evaluation/inference_samd.py \
  --fusion_mode naive \
  --samd_len_threshold 999999 \
  --bench_name mt_bench \
  ...
```

我会根据结果决定下一步诊断方向。
