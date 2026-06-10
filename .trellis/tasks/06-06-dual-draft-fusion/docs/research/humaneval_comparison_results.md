# HumanEval 对比实验结果分析

## 实验结果

```
Method              MAT     TPS      vs Eagle3   Steps
━━━━━━━━━━━━━━━━━━  ━━━━━━  ━━━━━━━  ━━━━━━━━━━  ━━━━━━
sam_sequence_graft  7.304   63.098   1.041x ✅   50.500
eagle3_only         6.948   60.606   1.000x      53.195
naive_fusion        5.915   49.481   0.816x ❌   62.433
```

## 核心结论

### ✅ 成功验证的假设

**假设 E：sam_sequence_graft ≠ naive_fusion**
- **证实**：sam_sequence_graft 比 Eagle3-only 快 4.1%
- **证实**：naive_fusion 比 Eagle3-only 慢 18.4%
- **结论**：问题在 naive_fusion 的实现，不是 HumanEval 不适合 SAM

### 🔍 根本原因分析

#### sam_sequence_graft (成功) vs naive_fusion (失败)

| 维度 | sam_sequence_graft | naive_fusion |
|------|-------------------|--------------|
| **融合策略** | Graft-style (条件串行) | Union-style (无条件并行) |
| **SAM 使用条件** | `match >= threshold` 才用 SAM | 无条件合并 SAM |
| **Eagle tree** | 保持完整 | 可能被 SAM 挤掉 |
| **融合方法** | `TreeSpec.graft_sequence()` | 自定义去重+排序 |
| **分数依据** | SAM 匹配长度（有意义） | depth proxy（不可靠） |

#### 为什么 sam_sequence_graft 成功？

```python
# sam_sequence_graft 的智能之处
if max(match_dyn, match_static) < threshold:
    # SAM 匹配差 → 只用 Eagle（避免低质量候选）
    return CandidateType.tree, pred_ids, eagle_buffers
else:
    # SAM 匹配好 → Graft 到 Eagle tree
    tree_spec = TreeSpec.from_eagle3_buffers(...).graft_sequence(sam_seq)
    return fused_tree
```

**关键**：
1. **质量过滤**：SAM 匹配差时不用，避免拉低整体质量
2. **保护 Eagle**：Eagle tree 优先级高，不会被 SAM 挤掉
3. **Graft 机制**：`graft_sequence()` 是专门设计的融合方法，比简单 union 更智能

#### 为什么 naive_fusion 失败？

```python
# naive_fusion 的问题
# 1. 无条件合并（即使 SAM 匹配差）
eagle_nodes = parse_eagle_tree(eagle_tree)
sam_nodes = parse_sam_sequence(sam_candidates)  # 不管质量如何

# 2. 分数融合不准确
eagle_score = 1.0 / (depth + 1)  # depth proxy
sam_score = match_length - position  # 意义不明（动态 SAM）

# 3. 去重可能误杀 Eagle 候选
merged = dedup(eagle_nodes, sam_nodes, strategy="max_score")
# 如果 SAM 分数被错误归一化得更高 → Eagle 被删除

# 4. 截断可能留下低质量 SAM
sorted_nodes = sort_by_normalized_score(merged)
selected = truncate(sorted_nodes, budget=60)
# 可能选中低质量 SAM，排除高质量 Eagle
```

**根本问题**：
- 缺少 **质量门控**（没有 threshold 判断）
- 分数融合 **不反映真实价值**（depth proxy 不可靠）
- **盲目 union**，没有保护 Eagle tree 的优先级

---

## 量化分析

### MAT 对比

```
sam_sequence_graft:  7.304  (+5.1% vs Eagle3)
eagle3_only:         6.948
naive_fusion:        5.915  (-14.9% vs Eagle3)
```

**解释**：
- sam_sequence_graft 每步多接受 **0.356 tokens**
- naive_fusion 每步少接受 **1.033 tokens**

**推算**：
```
# 假设 Eagle tree 有 42 个节点（典型值）
# sam_sequence_graft 添加 ~18 个 SAM 节点 → 60 总节点
# 这 18 个 SAM 节点平均贡献：0.356 / 18 ≈ 0.02 tokens/node

# naive_fusion 也添加 ~20-30 个 SAM 节点
# 但整体 MAT 下降 1.033
# → SAM 节点不仅没贡献，还挤掉了有用的 Eagle 节点
```

### Steps 对比

```
sam_sequence_graft:  50.5  (-5.1% vs Eagle3)
eagle3_only:         53.2
naive_fusion:        62.4  (+17.4% vs Eagle3)
```

**解释**：
- sam_sequence_graft 减少验证轮数 → 效率提升
- naive_fusion 增加验证轮数 → 效率下降

---

## 改进方案

### Phase 2.5：修复 naive_fusion（短期）

#### 方案 A：添加质量门控
```python
# 学习 sam_sequence_graft 的智能
def gen_candidates_naive_fusion(...):
    # 计算 SAM 匹配质量
    match_dyn, match_static = get_sam_match_quality(...)
    best_match = max(match_dyn, match_static)
    
    # 质量门控
    if best_match < threshold:
        # SAM 质量差 → 只用 Eagle
        return eagle_only_candidates
    
    # SAM 质量好 → 才进行融合
    return fuse_with_quality_sam(eagle, sam, best_match)
```

**预期收益**：MAT 提升到 6.5-6.8（接近 Eagle3-only）

#### 方案 B：保护 Eagle tree 优先级
```python
# 融合时给 Eagle 候选更高权重
def fuse_with_priority(eagle_nodes, sam_nodes):
    # Eagle 分数 × 1.5 (boost factor)
    for node in eagle_nodes:
        node.boosted_score = node.score * 1.5
    
    # 合并排序
    all_nodes = eagle_nodes + sam_nodes
    sorted_nodes = sort_by_boosted_score(all_nodes)
    
    # 确保至少保留 X% Eagle 节点
    selected = select_with_eagle_quota(sorted_nodes, eagle_ratio=0.7)
    return selected
```

**预期收益**：MAT 提升到 6.3-6.5

#### 方案 C：使用 TreeSpec.graft_sequence()
```python
# 直接复用 sam_sequence_graft 的融合机制
def gen_candidates_naive_fusion(...):
    eagle_tree = draft.tree_model.gen_draft(start_token)
    sam_seq = draft.sam.gen_draft_raw(...)
    
    # 使用 TreeSpec.graft_sequence()（已验证有效）
    tree_spec = TreeSpec.from_eagle3_buffers(...).graft_sequence(sam_seq)
    fused_buffers = tree_spec.to_buffers(...)
    
    return fused_tree
```

**优点**：
- 复用已验证的机制
- 代码简单
- 效果可预测

**缺点**：
- 与 sam_sequence_graft 重复
- 失去 "naive fusion" 作为 baseline 的价值

### Phase 3：Payoff-aware fusion（中期）

基于修复后的 naive_fusion，添加：

1. **提取 Eagle3 真实 logprob**（替换 depth proxy）
2. **Payoff 校准**（统一 Eagle/SAM 分数到 P(accept)）
3. **Context-aware weight**（根据任务类型动态调权）

**目标**：超过 sam_sequence_graft（> 1.041x Eagle3）

---

## 论文定位调整

### 新的故事线

**1. Motivation**
- Eagle3 和 SAM 应该互补
- 现有工作：Graft (串行), HVD (选择), sam_sequence_graft (条件串行)
- **我们**：真正的并行融合（无条件合并）

**2. Challenge**
- **Naive union 不够**：实验显示比 Eagle3-only 慢 18.4%
- 原因分析：
  - 无质量门控 → 低质量 SAM 候选拉低整体
  - 分数融合不准 → 误杀高质量 Eagle 候选
  - 缺少优先级 → Eagle tree 被破坏

**3. Our Solution: Payoff-aware fusion**
- P0: 提取 Eagle3 真实 logprob
- P1: 统一 Payoff 校准（P(accept) estimation）
- P2: Context-aware fusion weight
- 关键：在无质量门控的前提下，通过精准的 payoff 估计实现有效融合

**4. Results**
- Naive fusion: 0.816x ❌（证明简单融合不够）
- sam_sequence_graft: 1.041x ✅（现有最好方法）
- **Our payoff-aware**: > 1.05x ✅（目标）

**5. Contribution**
- 首个无条件并行融合（vs 条件串行）
- 统一 Payoff 框架（理论贡献）
- 证明：精准的 payoff 估计可以克服 naive union 的问题

### 与现有工作的对比

| Method | 策略 | 质量门控 | 分数融合 | Eagle 优先级 | 性能 |
|--------|------|---------|---------|------------|------|
| **Graft** | prune → graft | ✅ 剪枝 | N/A | ✅ 保护 | 好 |
| **HVD** | 选择 cache OR model | ✅ 预测接受长度 | N/A | ✅ 互斥 | 好 |
| **sam_sequence_graft** | 条件串行 | ✅ threshold | ✅ match_length | ✅ 保护 | 好 (1.041x) |
| **naive_fusion** | 无条件并行 | ❌ | ❌ depth proxy | ❌ | 差 (0.816x) |
| **payoff-aware (我们)** | 无条件并行 | ✅ payoff threshold | ✅ 校准到 P(accept) | ✅ weight | 目标 > 1.05x |

---

## 下一步行动

### 立即执行（今天）

**任务 1：修改 naive_fusion 输出详细统计**
- Eagle/SAM 各自接受率
- SAM 平均匹配长度
- 双引擎一致性
- 每个样本的融合 metadata

**任务 2：实现 Phase 2.5 快速修复**
- 优先级：方案 A（质量门控）> 方案 B（Eagle 优先级）
- 目标：将 naive_fusion 提升到 0.9x 以上

### 本周完成

**任务 3：提取 Eagle3 真实 logprob**
**任务 4：实现基础 Payoff 校准**

### 目标

- Phase 2.5 修复后：naive_fusion > 0.9x Eagle3
- Phase 3 完成后：payoff-aware > 1.05x Eagle3（超过 sam_sequence_graft）

---

## 实验数据文件

```
远端路径：
/teamspace/studios/this_studio/SAM-Decoding/

文件：
- humaneval_sam_graft.jsonl          (7.304 MAT, 63.098 TPS)
- humaneval_eagle_only.jsonl         (6.948 MAT, 60.606 TPS)
- humaneval_naive_fusion.jsonl       (5.915 MAT, 49.481 TPS)
- logs/humaneval_fusion_20260607_031615/summary.txt
```

这些数据对论文至关重要，建议：
1. 备份到本地
2. 分析典型样本的融合细节
3. 可视化融合树对比
