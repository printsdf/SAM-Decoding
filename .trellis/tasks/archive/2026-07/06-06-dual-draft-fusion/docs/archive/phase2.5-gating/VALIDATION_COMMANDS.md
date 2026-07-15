# Phase 2.5 Part 2 验证命令（质量门控已实现）

## ✅ Code Review 结果

**修改文件**: `samd/utils.py`

**关键实现**:
1. ✅ 添加了 `_extract_candidate_tokens_eagle()` 辅助函数
2. ✅ 在 `fusion_mode == "naive"` 分支添加 SAM 质量检查
3. ✅ 质量门控：`if best_match < threshold: return Eagle-only`
4. ✅ 统计记录完整：`sam_skipped=True`, `sam_match_quality`
5. ✅ 返回格式正确：与融合分支一致
6. ✅ 编译检查通过

**代码质量**: 优秀，实现清晰，逻辑正确

---

## 🚀 下一步：远端验证

### Step 1: 同步代码到远端

```bash
# 在远端
cd /teamspace/studios/this_studio/SAM-Decoding

# 方式 1: 如果用 git
git pull origin feature/eagle3-tail-sidecar

# 方式 2: 或直接复制修改的文件
# 从本地复制 samd/utils.py 到远端
```

### Step 2: 快速验证（10 样本）

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 运行 10 样本测试
python evaluation/inference_samd.py \
  --model_type llama \
  --model_path /teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
  --model_id llama31-8b-instruct \
  --tree_model_path /teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
  --bench_name humaneval \
  --question_begin 0 \
  --question_end 10 \
  --num_choices 1 \
  --max_new_tokens 512 \
  --dtype bfloat16 \
  --fusion_mode naive \
  --fusion_dedup_strategy max_score \
  --fusion_max_draft_tokens 60 \
  --answer_file humaneval_naive_gated_test_10.jsonl \
  2>&1 | tee humaneval_naive_gated_test_10.log

# 查看统计（在日志末尾）
tail -50 humaneval_naive_gated_test_10.log
```

### Step 3: 检查关键指标

**期望改进**:
```
修改前（无质量门控）:
- Eagle accept: 13.74% ❌
- SAM accept: 10.04%
- Mean accept: 5.798

修改后（有质量门控）:
- Eagle accept: > 50% ✅ (目标)
- SAM skipped: ~70-80% (大部分情况 SAM 质量差)
- Mean accept: > 6.5 (目标 > 0.9x Eagle3-only 的 6.948)
```

**判断标准**:
- ✅ 如果 Eagle accept > 50%：质量门控有效，继续 Step 4
- ⚠️ 如果 Eagle accept 仍 < 40%：需要调整 threshold 或叠加方案 B
- ❌ 如果更差：检查代码是否有问题

### Step 4: 完整评估（如果 Step 3 通过）

```bash
# 完整 HumanEval 164 样本
python evaluation/eval_naive_fusion.py \
  --model_path /teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
  --tree_model_path /teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
  --bench_name humaneval \
  --num_choices 1 \
  --answer_dir naive_fusion_gated_20260607

# 等待完成后查看结果
cat logs/naive_fusion_gated_20260607/summary.txt
```

**目标**:
```
当前基准:
- eagle3_only: 6.948 MAT, 60.606 TPS (1.000x)
- sam_sequence_graft: 7.304 MAT, 63.098 TPS (1.041x)
- naive_fusion (无门控): 5.915 MAT, 49.481 TPS (0.816x)

目标（有门控）:
- naive_fusion_gated: > 6.5 MAT, > 54 TPS (> 0.9x) ✅
```

---

## 📊 统计分析要点

查看 `Naive Fusion Analysis` 输出，关注：

1. **SAM skipped 比例**
   - 预期：70-80%（大部分情况 SAM 质量差）
   - 如果 < 50%：threshold 可能太低

2. **Eagle accept rate**
   - 预期：> 50%（接近 Eagle-only）
   - 如果 < 40%：质量门控效果不佳

3. **SAM contribution（融合时）**
   - 预期：> 20%（只在质量好时融合）
   - 如果 < 15%：即使通过门控，SAM 仍贡献不足

4. **Mean accept 提升**
   - 预期：从 5.798 提升到 > 6.5
   - 如果 < 6.0：可能需要方案 B（Eagle boost）

---

## 🔧 调试建议（如果结果不理想）

### 问题 1: SAM skipped 比例太低（< 50%）

**原因**: threshold 太低，让太多低质量 SAM 通过

**解决**: 提高 threshold
```bash
# 尝试不同 threshold
python evaluation/inference_samd.py \
  --samd_len_threshold 3 \  # 默认可能是 2
  --fusion_mode naive \
  ...
```

### 问题 2: Eagle accept 仍然低（< 40%）

**原因**: 即使加了门控，融合分支仍有问题

**解决**: 叠加方案 B（Eagle 优先级）

修改 `samd/samd_config.py` 添加配置：
```python
fusion_eagle_boost: float = 1.2
fusion_min_eagle_ratio: float = 0.7
```

然后修改融合代码使用这些参数。

### 问题 3: 整体性能没提升

**原因**: SAM 在 HumanEval 上本身不适合

**策略调整**:
- 论文中报告：质量门控改善了稳定性
- 但 naive fusion 仍不如 sam_sequence_graft
- 为 Phase 3 (payoff-aware) 提供动机

---

## ✅ 成功标准

**Phase 2.5 完成**:
- [x] 统计代码实现
- [x] 质量门控实现
- [x] 代码 review 通过
- [ ] 10 样本验证（远端执行中）
- [ ] 完整评估（待 10 样本通过）
- [ ] 性能提升到 > 0.9x Eagle3

**下一步**:
- 如果成功：准备 Phase 3（提取 Eagle3 logprob + Payoff 校准）
- 如果不理想：分析数据，调整参数或叠加方案 B
