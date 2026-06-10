# 相关工作调研

## 1. Graft: Draft Less, Retrieve More (arXiv:2605.20104v1, 2026-05)

### 核心思路
"prune-then-graft" 策略：动态剪枝释放预算 → 检索填充空位

### 关键机制
1. **动态剪枝**：在置信度检查点剪枝低信心分支
2. **检索补偿**：用 GPU-resident adjacency matrix 检索候选
3. **固定预算**：总验证 token 数不变（如 60），只改变组成
4. **Root-centered retrieval**：从当前 root token 并行检索
5. **在线更新**：用 target model 验证结果更新检索矩阵

### 实验结果
- Short-context: 最高 5.41× speedup (Vicuna-13B HumanEval)
- 相比 EAGLE3 平均提升: 7.4%-21.8% (模型越大提升越明显)
- Long-context: 3.22× average speedup (LLaMA3.1-8B)

### 与我们的区别
| 维度 | Graft | 我们 |
|------|-------|------|
| 时序 | 串行（先剪枝后检索） | 并行（同时生成后融合） |
| 检索角色 | 补救措施（填补剪枝损失） | 平等伙伴（与 Eagle 共同贡献） |
| 预算分配 | 剪枝阶段决定 | 动态权重决定 |
| 分数融合 | 无需融合（不竞争） | 需要校准融合 |
| 插入位置 | Root-centered | Tree-aware（关键节点） |

**结论**：Graft 把检索当"补救"，我们把检索当"伙伴"。本质差异是并行 vs 串行。

---

## 2. HVD: Hybrid Verified Decoding (arXiv:2606.01019v1, 2026-05)

### 核心思路
Payoff-guided drafter selection：预测 cache draft 的接受长度 → 选择 cache OR model draft

### 关键机制
1. **接受长度预测**：训练模型预测 `E[accepted_length | cache_draft]`
2. **阈值选择**：`if predicted_length >= τ then use_cache else use_eagle3`
3. **单一 drafter**：每步只验证一个 draft source
4. **Agentic workflow 优化**：在结构化任务中效果显著

### 实验结果
- Agentic workflows: 2.73× average speedup，在所有设置下超过 EAGLE3
- Cache 提供长 continuation 但收益不稳定
- 预测模型能有效避免低收益的 cache 验证

### 与我们的区别
| 维度 | HVD | 我们 |
|------|-----|------|
| 核心机制 | 选择（cache OR model） | 融合（cache AND model） |
| 决策单元 | 整个 draft 的预测接受长度 | 候选节点级别的 payoff |
| 验证输入 | 单一 drafter 输出 | 融合后的树 |
| 目的 | 避免低收益验证 | 最大化多源候选价值 |
| 学习信号 | 接受长度预测 | 置信度校准 + 融合权重 |

**结论**：HVD 是 routing policy（路由），我们是 fusion policy（融合）。本质差异是选择 vs 合并。

---

## 3. Codex 技术评估总结

### 提出的核心问题

**问题 1: 去重语义错误**
- 错误做法：相同 token 字符串去重
- 正确做法：只有 (parent_path, token) 相同才能合并
- **解决方案**：用 Trie 结构，键为 (path, token)

**问题 2: 分数不可直接比较**
- Eagle 置信度：模型概率，已校准到接受率
- SAM 匹配分数：依赖语料域、重复性、结构
- **解决方案**：统一校准到 `P(accept)` → 计算 expected payoff

**问题 3: 简单 max 丢失证据**
- 双引擎同时提出 → 强信号
- 只保留 max → 丢失一致性信息
- **解决方案**：保留双源，加 agreement bonus

**问题 4: 全局截断破坏树结构**
- 子节点需要祖先存在
- 按单节点分数排序 → 可能选中孤儿节点
- **解决方案**：前缀闭包约束，计算 path value

**问题 5: 固定上限可能反效果**
- max_draft=5/8 太小 → 浪费机会
- 草稿过少 → 增加验证轮次
- **解决方案**：动态预算，根据场景调整

### 推荐的技术方案

**方案：校准的、有预算的、前缀闭合融合树**
```python
1. 合并到 Trie: 键为 (parent_path, token)
2. 保留源特征: Eagle logprob, SAM match_length, 一致性
3. 校准到 P(accept): isotonic regression
4. 计算路径价值: V(node) = V(parent) * P(accept)
5. 预算约束选择: 保留祖先，贪心选高 path_value
6. 动态预算控制: 根据接受率调整
```

**或采用 prune-then-graft 变体**：
先 Eagle 树 → SAM 填补高价值空缺（更清晰）

### 需要跟踪的指标
```python
# 核心指标
- tokens/sec, MAT, verified_per_accepted

# 来源分析  
- eagle_only_accept, sam_only_accept, both_accept
- duplicate_rate, agreement_accept_rate

# 校准质量
- calibration_curves, ECE, Brier_score

# 预算利用
- draft_budget_utilization, marginal_utility

# 场景回归
- by_workload: code, RAG, summarization, math, long_context
- by_temperature: greedy vs stochastic
```

---

## 4. 创新性定位

### 与现有工作的差异矩阵

| 工作 | 机制 | Eagle | SAM | 时序 | 预算 | 分数处理 |
|------|------|-------|-----|------|------|---------|
| **Graft** | prune→graft | 剪枝后保留 | 填充空位 | 串行 | 剪枝释放 | 无需融合 |
| **HVD** | selection | 可选 | 可选 | 互斥 | 单源全部 | 预测接受长度 |
| **我们** | fusion | 同时生成 | 同时生成 | 并行 | 动态分配 | 校准后融合 |

### 核心创新点

**1. 首个真正的双引擎并行融合**
- Graft 是串行补救
- HVD 是互斥选择
- 我们是并行融合

**2. Context-aware fusion policy**
- 根据任务类型、上下文长度、token 熵动态调权
- 理论：不同场景下 Eagle/SAM 优势不同
- 实践：code 偏 SAM，creative 偏 Eagle

**3. Tree-aware insertion strategy**
- 不是简单的 root-centered（Graft）
- 根据 Eagle 树结构选择插入位置
- 高置信度 → sibling extension
- 低置信度 → fallback path

**4. 统一的 payoff estimation framework**
- 将 Eagle 置信度和 SAM 分数统一到 `E[accepted_tokens]`
- 在线校准，适应不同场景
- 理论支撑：expected payoff = P(accept) × depth

---

## 5. 论文定位建议

### Title
"Payoff-Guided Dual Draft Fusion for Speculative Decoding"

### Abstract 关键句
> While recent work explores either sequential compensation (Graft) or runtime selection (HVD) between model-based and retrieval-based drafters, we propose the first parallel fusion framework that treats both sources as equal partners. Our method dynamically calibrates their contributions via context-aware fusion weights and tree-aware insertion strategy.

### 核心 claims
1. **首个并行融合方法**（区别于串行和选择）
2. **Context-aware 优于固定规则**（消融实验证明）
3. **Tree-aware 优于 root-centered**（在 code 和 long-context 场景）
4. **统一 payoff 框架**（理论贡献）

### 实验对比必须包括
- Graft (prune-then-graft baseline)
- HVD (selection baseline)
- Naive fusion (our method ablation)
- Our full method

### 预期结果
- 在 code generation 和 long-context 场景下显著优于 Graft 和 HVD
- 消融实验证明 context-aware 和 tree-aware 的有效性
- 跨场景分析：解释何时融合优于选择/串行

---

## 6. 风险与对策

### 风险 1: 创新性被质疑"只是工程优化"
**对策**：
- 强化理论框架：payoff estimation 的数学建模
- 深入分析：为什么并行融合在某些场景下优于串行/选择
- 泛化能力：框架适用于其他 drafter 组合

### 风险 2: 实验结果不显著
**对策**：
- 聚焦优势场景：code generation, long-context
- 分析 failure cases：何时融合不如选择
- 提供 selection policy：何时应该用融合 vs 选择

### 风险 3: 实现复杂度高
**对策**：
- 提供简化版本：naive fusion 作为实用 baseline
- 开源实现：易集成到现有系统
- 性能优化：GPU-resident + 并行化

---

## 下一步行动

### 立即开始（本周）
1. 实现 naive fusion baseline
2. 在 HumanEval 上验证可行性
3. 对比 Eagle3-only, SAM-only, naive fusion

### 短期目标（2周内）
1. 实现 payoff-aware fusion
2. 添加在线校准
3. 扩展到更多数据集

### 中期目标（1个月内）
1. 实现 tree-aware insertion
2. 完整对比实验（vs Graft, HVD）
3. 消融实验
4. 论文初稿

### 长期目标
1. 投稿 ACL
2. 开源实现
3. 集成到主流推理框架
