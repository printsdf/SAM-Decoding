# 新研究方向探索 - Codex + Gemini 双模型建议

## 生成日期
2026-06-10

## 背景

Depth-Decoupled Extension 在两个数据集上均未达标：
- MedQA: oracle gap = 0.00%
- MT-Bench: oracle gap = 0.61% (< 3% 决策门)

**核心教训**：问题不在融合策略，而在 SAM 使用范式

---

## Codex 建议（3个方向）

### 🏆 #1: Rejection-Boundary SAM Repair ⭐⭐⭐⭐⭐

**核心思想**：在 EAGLE 首次拒绝处使用 SAM 救援，而非在成功后扩展

**机制**：
1. 预测 EAGLE 可能的首次拒绝深度（用 confidence/entropy/margin）
2. 从该 prefix 查询 SAM
3. 仅在拒绝边界处嫁接少量 SAM 修复分支

**与 depth-decoupled 的区别**：
- Depth-decoupled: SAM 扩展 EAGLE 的成功部分
- Rejection-boundary: SAM 修复 EAGLE 的失败点
- **范式转换**：从"协作"到"救援"

**预期 oracle gap**：6-9%（如果 15-25% 步骤有早期拒绝且 SAM 能救援 1/3）

**Oracle 可测性**：
- 第一步：用真实拒绝深度作弊测试（理想情况）
- 第二步：用 EAGLE confidence 预测拒绝深度
- 如果第一步都失败，方向直接放弃

**实现复杂度**：Medium
- 需要：拒绝深度日志、边界选择器、SAM 查询、树合并

**关键风险**：EAGLE confidence 可能无法准确预测拒绝位置，或 SAM 在修正后的 prefix 上匹配弱

---

### #2: Complementary Root-Budget Substitution ⭐⭐⭐

**核心思想**：将 EAGLE 的低效分支预算重新分配给 SAM

**机制**：
1. 保持总验证预算不变
2. 识别 EAGLE 树中从未被接受的节点（零效用分支）
3. 用 SAM 替换这些分支（仅当 SAM 有强证据：长匹配、prompt 局部匹配等）

**与 depth-decoupled 的区别**：
- Depth-decoupled: EAGLE 59 节点固定，SAM 在后面
- Root-budget: EAGLE 预算可重分配，SAM 覆盖不同模式

**预期 oracle gap**：5-7%（如果少量零接受/单 token 接受的 cycle 能在 depth 0-2 被救援）

**实现复杂度**：Medium
- 需要：node-level oracle accounting、确定性替换策略、固定预算树构建

**关键风险**：EAGLE 近根分支可能已经接近最优，没有冗余预算给 SAM

---

### #3: Episodic High-Precision SAM ⭐⭐⭐⭐

**核心思想**：用高精度 memory 替换通用 corpus，提升 SAM 质量

**机制**：
1. 停止使用 SAM 作为通用 corpus matcher
2. 构建源标记的 SAM memory：
   - Prompt spans
   - 当前对话
   - 之前请求的已接受输出
   - Benchmark train split 输出
   - 领域特定生成 trace
3. 按 match length 和 source type 门控 SAM 使用
4. 仅嫁接长"爆发"延续（10+ tokens）

**与 depth-decoupled 的区别**：
- Depth-decoupled: 改融合几何
- High-precision: 改检索基底（attack quality directly）

**预期 oracle gap**：6-12% (on high-match slices)；整体 >5% 需要这些 slice 覆盖足够流量

**实现复杂度**：Low-Medium
- 需要：memory 构建、source tagging、match-length gating、oracle bucket 分析

**关键风险**：
- MT-Bench/MedQA 的精确 suffix 重叠可能太稀疏
- 如果 memory 从 eval answers 构建，注意 benchmark leakage

---

## Gemini 建议（3个方向）

### #1: Entropy-Aware Adaptive Switching (E-AAS) ⭐⭐⭐⭐

**核心假设**：EAGLE 在"决策点"（高熵）最容易失败，SAM 的精确检索能提供锚定路径

**机制**：
1. 监控 EAGLE depth-1 的 confidence
2. 如果 P(EAGLE_top1) < τ（如 0.5），识别为高熵步骤
3. 切换预算：不用宽 EAGLE 树，改用 **SAM-First** 搜索

**Oracle 可测性**：
- 过滤现有 trace 中 EAGLE top-1 logprob < threshold 的步骤
- 比较这些步骤上 SAM 候选 vs EAGLE 候选的接受率

**预期 oracle gap**：5-7% (MedQA/GSM8K，精确术语/逻辑比通用流畅性更重要的任务)

**实现复杂度**：Medium
- 需要：定义鲁棒的、硬件感知的切换阈值 τ

**关键挑战**：τ 触发太频繁会损失 EAGLE 的"简单"收益

---

### #2: Structural Macro-Grafting (SMG) ⭐⭐⭐⭐⭐

**核心假设**：EAGLE 是局部 k-gram 预测器（horizon 4-6 tokens），无法捕捉 10-20+ token 的结构重复（JSON schema、boilerplate code、重复引用）

**机制**：
1. SAM 作为独立的"长射程"生成器并行运行
2. 如果 SAM 找到 cache match > L（如 L=12），产生单个深"刺探"候选
3. 与宽浅 EAGLE 树一起验证
4. 从"平衡树"变为"卫星树"（EAGLE bush + SAM needle）

**Oracle 可测性**：
- 搜索 `sam_sequence_graft` trace
- 找 base model 接受了 10+ token 连续序列，且在 SAM cache 中存在，但在 EAGLE 树中被分散或错过的情况

**预期 oracle gap**：10-15% (Code/RepoCoder 和 Long-Context RAG 任务)

**实现复杂度**：Medium-High

**关键挑战**：
- 验证预算管理
- 20-token SAM spike 是"all-or-nothing"
- 如果在 token 1 失败，剩余 19 token 的预算浪费

---

### #3: SAM-Guided Tree Diversification (SGDE) ⭐⭐⭐

**核心假设**：EAGLE 树搜索是"mode-seeking"，探索最可能预测 prefix 的变体。但 base model 常接受不是 top MLP prediction 的"检索密集型" token。

**机制**：
1. 用 SAM 种子 EAGLE 树
2. 不从 EAGLE top-4 tokens 开始树，改用 `[EAGLE_top1, EAGLE_top2, SAM_top1, SAM_top2]`
3. 强制树搜索从根部同时探索"检索路径"和"预测路径"

**Oracle 可测性**：
- 比较同样大小的树：
  - 从 `EAGLE[1:k]` 构建 vs
  - 从 `EAGLE[1:k-1] + SAM[1]` 构建
- 的 oracle ceiling

**预期 oracle gap**：4-6% (General Chat / MT-Bench)

**实现复杂度**：Medium-High
- 需要修改 tree-building kernel（如 `TopKEngine` 或 EAGLE3 draft loop）
- 支持多个分歧 root seeds 而不增加 latency

**关键挑战**：集成复杂度高

---

## 共识分析

### 🎯 强共识：Rejection-Boundary 范式

**Codex #1** (Rejection-Boundary SAM Repair) 和 **Gemini #1** (Entropy-Aware Switching) 都认同：

**SAM 应该在 EAGLE 失败/不确定处介入，而非在成功处扩展**

这是对 depth-decoupled 失败的最直接回应。

### 💎 独特高价值方向

**Gemini #2 (Structural Macro-Grafting)**：
- 唯一针对 **长程结构重复** 的方向
- Oracle gap 预期最高：10-15%
- 适合 Code/RAG 任务，与其他方向正交
- 实现风险最高（all-or-nothing）

---

## 推荐行动计划

### Phase A: 快速 Oracle 验证（1-2 天）

**并行验证两个最有潜力方向**：

#### 1. Rejection-Boundary Recovery (Codex #1)

**Oracle 测试**：
```python
# 分析现有 trace
for step in trace:
    first_reject_depth = find_first_reject(step.eagle_tree, step.accepted)
    if first_reject_depth is not None:
        sam_rescue_candidates = [c for c in step.sam_candidates
                                 if c.depth == first_reject_depth]
        # 计算 SAM 能否救援
```

**决策门**：条件 oracle gap > 7% → 实现

#### 2. Episodic High-Precision SAM (Codex #3)

**Oracle 测试**：
```python
# 按 match_length 分层
for step in trace:
    high_quality_sam = [c for c in step.sam_candidates
                       if c.match_length >= 10]
    # 重新计算 oracle gap（仅高质量 SAM）
```

**决策门**：High-match slice gap > 8% → 实现

### Phase B: 实现最佳方向（1 周）

- **Rejection-Boundary 胜** → 实现完整版（Medium 难度）
- **High-Precision SAM 胜** → 重建 memory + gating（Low-Medium 难度）
- **两个都高** → 组合使用

### Phase C: 备选激进方向

如果 Phase A 两个都失败：

**Structural Macro-Grafting** (Gemini #2)
- 在 Code 数据集（RepoCoder/HumanEval）上测试
- 高风险高回报，oracle gap 预期 10-15%
- 适合作为独立论文方向

---

## 下一步立即行动

1. ✅ 创建 `rejection_boundary_oracle.py` 分析脚本
2. ✅ 创建 `high_precision_sam_oracle.py` 分析脚本
3. 在现有 MT-Bench trace 上运行两个 oracle 分析
4. 根据结果决定实现方向

**预计时间**：2-3 天完成 Phase A oracle 验证
