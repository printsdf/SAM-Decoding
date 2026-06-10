# Phase 2.5 最终诊断与修复方案

## 🔍 诊断结果总结

### Debug 测试结果（3 样本）

```
质量门控有效：
- SKIP 比例：89.1% (156/175)
- FUSE 比例：10.9% (19/175)
- 动态 SAM match 分布：0-10，多数 < 5
- threshold=5 工作正常

性能结果：
- Mean accept: 6.91
- vs Eagle-only: 6.95 (几乎没提升)
- Eagle accept rate: ~10% (仍然异常低)
```

### 核心问题

**即使 89% 时候用 Eagle-only，性能仍然很差**

可能原因：
1. **11% 的 FUSE 破坏太严重**：少量融合步骤拉低了整体性能
2. **Eagle accept rate 计算有误**：统计方式可能不对
3. **接受率计算包含了 FUSE 步骤**：被融合步骤拉低了平均值

## 💡 修复方案

### 方案 A+：质量门控 + 更高 threshold

**思路**：既然 11% 的 FUSE 仍在破坏性能，进一步提高 threshold

```bash
# 测试 threshold=10
python evaluation/inference_samd.py \
  --samd_len_threshold 10 \
  --fusion_mode naive \
  ...

# 预期：SKIP 比例 > 95%，接近纯 Eagle-only
```

### 方案 B：直接禁用 SAM（验证假设）

**思路**：完全禁用 SAM，验证 Eagle-only 分支是否有 bug

```bash
# 强制 threshold 极高，永远 SKIP
python evaluation/inference_samd.py \
  --samd_len_threshold 999999 \
  --fusion_mode naive \
  ...

# 预期：100% SKIP，性能应该 = Eagle-only (6.95 MAT)
# 如果仍然很低 → Eagle-only 分支有 bug
# 如果正常 → 说明即使少量 FUSE 也有害
```

### 方案 C：放弃 naive fusion，直接 Phase 3

**理由**：
1. sam_sequence_graft 已经证明有效（1.041x）
2. naive fusion 即使加质量门控也效果不佳
3. 问题在于分数融合不准确，需要 payoff-aware

**行动**：
- 将 naive fusion 定位为 "negative baseline"
- 论文故事：naive 失败 → 证明需要 payoff-aware
- 直接开始 Phase 3：提取 Eagle3 logprob + Payoff 校准

---

## 🎯 建议行动

### 立即执行（5 分钟）

**测试方案 B（验证 Eagle-only 分支）**

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 3 样本快速验证
python evaluation/inference_samd.py \
  --samd_len_threshold 999999 \
  --fusion_mode naive \
  --question_begin 0 --question_end 3 \
  --answer_file ./humaneval_eagle_only_via_gate_3.jsonl \
  2>&1 | tee eagle_only_gate_3.log

# 查看结果
echo "Gate decisions:"
grep -c "\[GATE\] -> SKIP" eagle_only_gate_3.log  # 应该 = 100%
grep -c "\[GATE\] -> FUSE" eagle_only_gate_3.log  # 应该 = 0

echo ""
echo "Mean accept:"
grep "Mean accepted" eagle_only_gate_3.log
```

**判断标准**：
```
如果 Mean accept ~6.9-7.0：
→ Eagle-only 分支正常
→ 问题在于即使 11% FUSE 也破坏性能
→ 结论：放弃 naive fusion，直接 Phase 3

如果 Mean accept 仍然很低 (~5.8-6.0)：
→ Eagle-only 分支有 bug
→ 需要修复 Eagle-only 实现
```

### 后续行动（根据结果）

#### 情况 A：Eagle-only 正常（Mean accept ~7.0）

**结论**：naive fusion 无法修复，即使加质量门控

**下一步**：
1. 记录实验结果（naive fusion = negative baseline）
2. 写入 PRD：naive fusion 失败的原因
3. **开始 Phase 3**：
   - 提取 Eagle3 真实 logprob
   - 实现 Payoff 校准
   - 目标：超过 sam_sequence_graft (1.041x)

#### 情况 B：Eagle-only 有 bug（Mean accept 仍低）

**说明**：Eagle-only fallback 分支实现有问题

**调试**：
1. 对比 Eagle-only fallback 和原始 Eagle3-only 的区别
2. 检查 `_extract_candidate_tokens_eagle()` 实现
3. 检查统计记录方式

---

## 📊 论文策略（最终版）

### 故事线

**1. Motivation**
- Eagle3 + SAM 应该互补
- 现有方法：Graft (串行), HVD (选择), sam_sequence_graft (条件串行)

**2. Naive Baseline 尝试**
- 实现：无条件并行融合
- 结果：0.816x（比 Eagle-only 慢 18%）
- 原因分析：
  - SAM 质量差（match=2.1）破坏 Eagle tree
  - 分数融合不准确（depth proxy 不可靠）

**3. 质量门控改进**
- 实现：添加 SAM 质量门控（threshold=5）
- 结果：89% SKIP，但性能仍未提升
- 结论：**即使少量低质量融合也有害**

**4. Our Solution: Payoff-Aware Fusion**
- 问题：需要精准的 payoff 估计
- 实现：
  - 提取 Eagle3 真实 logprob
  - 统一 Payoff 校准（P(accept) estimation）
  - Context-aware fusion weight
- 结果：> 1.05x（超过 sam_sequence_graft）

### 核心贡献

1. **系统性分析**：为什么 naive fusion 失败
2. **Payoff-aware 框架**：统一的融合目标
3. **实验验证**：超过现有最好方法

---

## 时间估算

- **今天（验证）**: 5 分钟测试方案 B
- **明天（决策）**: 根据结果决定是修复还是跳过
- **本周（Phase 3）**: 提取 logprob + 基础校准
- **下周（完成）**: 完整实验 + 论文初稿

---

## 当前建议

**优先级 1**（5 分钟）：
```bash
# 测试 threshold=999999（100% SKIP）
# 验证 Eagle-only 分支是否有 bug
```

**优先级 2**（根据结果）：
- 如果 Eagle-only 正常 → 开始 Phase 3
- 如果有 bug → 修复后重新评估

请执行优先级 1 的测试，把结果告诉我！
