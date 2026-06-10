# Depth-Decoupled Dual Draft Fusion for Speculative Decoding

## 决策变更记录

**2026-06-10: 方向调整 - Phase 3 Payoff-Aware → Depth-Decoupled Extension**

**原计划（Phase 3 Payoff-Aware Fusion）**：
- 提取 Eagle3 真实 logprob ✅ 已完成
- 实现统一 Payoff 校准器
- Context-aware fusion weight

**变更原因**：
- Codex + Gemini 双模型共识推荐 **Depth-Decoupled Extension**
- 创新性更强：尺度解耦（syntax vs semantics），层次化分工
- 实现成本可控：3-5 天，复用现有 TreeSpec API
- Oracle gap 预期：5-8% vs Eagle3-only

**新方向（Depth-Decoupled Extension）**：
- **核心思想**：Eagle 负责浅层（depth 0-D_split），SAM 仅从 Eagle 叶节点扩展深层
- **Phase A（当前）**：Oracle 分析 - 寻找最优 D_split 切换点
- **Phase B**：实现 `tree_fusion="eagle_leaf_sam_extend"`
- **决策门**：Oracle gap > 5% → 实现完整方法；否则考虑其他方向

**已完成工作**：
- ✅ Codex 实现 `oracle_depth_decoupled.py` 分析工具
- ✅ 集成到 `oracle_fusion_analysis.py`
- ✅ 单元测试通过

**下一步**：运行 Oracle 分析（需要远程 GPU 环境 + 现有 fusion profile trace）

---

## 问题定位

**现有工作的局限**：
- **Graft (arXiv:2605.20104)**: prune-then-graft，串行机制，剪枝后用检索补救
- **HVD (arXiv:2606.01019)**: 选择模式，每步选择 cache OR model draft
- **现有 SAM-Decoding**: routing 模式，根据匹配质量决定是否用 SAM

**核心机会**：真正的双引擎并行融合 —— Eagle3 AND SAM 同时工作，候选级别融合

## 研究目标

设计并实现一个新颖的双草稿树融合算法，作为 ACL 论文的核心模块，目标创新点：

1. **并行融合机制**（区别于 Graft 的串行、HVD 的选择）
2. **Context-aware fusion policy**（动态权重调整）
3. **Tree-aware insertion strategy**（智能插入位置选择）
4. **统一的 payoff estimation framework**（理论支撑）

## 技术方案

### Phase 1: 理论框架设计

**1.1 Expected Payoff 统一建模**
- 目标：将 Eagle 置信度和 SAM 匹配度统一到 `E[accepted_tokens]`
- Eagle: `P_accept(logprob, depth) * path_depth`
- SAM: `P_accept(match_length, frequency, domain) * path_depth`
- 校准方法：isotonic regression on verification history

**1.2 Context Feature Extraction**
```python
context = {
    "token_entropy": H(p_next),           # 当前 token 熵
    "suffix_match_quality": SAM.score,    # SAM 匹配质量
    "domain_type": detect_domain(),       # code/chat/summarization
    "context_length": len(history),       # 上下文长度
    "recent_accept_rate": avg(last_N),    # 近期接受率
}
```

**1.3 Fusion Weight Function**
```python
def compute_weights(context):
    w_eagle = 0.5
    w_sam = 0.5
    
    # 代码生成 -> SAM 权重高
    if context.domain == "code": w_sam += 0.2
    
    # 高熵 -> Eagle 权重高
    if context.entropy > θ_high: w_eagle += 0.2
    
    # 长上下文 -> SAM 权重高
    if context.length > 4096: w_sam += 0.1 * (length/4096)
    
    # SAM 匹配差 -> Eagle 权重高
    if context.sam_quality < θ_low: w_eagle += 0.3
    
    return normalize(w_eagle, w_sam)
```

### Phase 2: Naive Baseline 实现 ✅

**状态**: 已完成实现与实验

**2.1 简单合并策略**
- Eagle3 生成 tree，SAM 生成 sequence
- 合并规则：相同 token 保留最高置信度（`max_score`）
- 固定预算：max_draft = 60，超限按 score 截断
- 分数归一化：Eagle (depth proxy) 和 SAM (match_length) 分别归一化

**2.2 实现完成**
- ✅ 创建 `samd/fusion/` 模块（437 行核心代码）
- ✅ 修改 `samd/utils.py::gen_candidates()` 添加 `fusion_mode="naive"` 分支
- ✅ 完整的统计输出：eagle_nodes, sam_nodes, accept_rates, contribution
- ✅ 单元测试通过（9/9）

**2.3 实验结果（HumanEval）**

| Method | MAT | TPS | Speedup | Steps |
|--------|-----|-----|---------|-------|
| **sam_sequence_graft** | **7.304** | **63.098** | **1.041x** | 50.5 |
| eagle3_only | 6.948 | 60.606 | 1.000x | 53.2 |
| **naive_fusion** | **5.915** | **49.481** | **0.816x** ❌ | 62.4 |

**关键发现**：
- ❌ Naive fusion 比 Eagle3-only 慢 18.4%
- ✅ sam_sequence_graft 仍然最好（1.041x）
- ⚠️ 问题诊断：
  - SAM avg match=2.1（质量差）
  - Eagle accept rate=13.74%（应该 60-70%）
  - SAM 占 41% 候选但贡献仅 18%

**2.4 Phase 2.5: 质量门控尝试** ✅

**实施**: 添加 SAM 质量门控（`if match < threshold: skip SAM`）

**结果（Debug 测试，3 样本）**:
- ✅ 门控有效触发：89.1% SKIP (156/175)
- ✅ Eagle-only 分支正常：Mean accept=6.72
- ❌ 性能仍未提升：即使 89% 用 Eagle-only，整体性能仍低

**结论**: 
- **Naive fusion 根本性失败**：分数融合不准确是核心问题
- **质量门控无法修复**：需要精准的 payoff 估计
- **直接进入 Phase 3**：提取真实 logprob + payoff 校准

**2.5 主要数据集调整**
- ❌ HumanEval：静态 SAM 会数据泄露，动态 SAM 质量差
- ✅ **MT-Bench**：已有成功案例（sam_sequence_graft=1.044x），作为主要验证集
- ✅ MedQAd：作为跨域验证

### Phase 3: Payoff-Aware Fusion（当前重点）

**目标**: 通过精准的 payoff 估计实现有效融合，超过 sam_sequence_graft (1.044x)

**3.0 核心问题**
- Naive fusion 失败原因：Eagle depth proxy 不可靠，无法准确估计 P(accept)
- 解决思路：提取真实 Eagle3 logprob + 统一 Payoff 校准框架

**3.1 P0: 提取 Eagle3 真实 logprob** ⭐

**当前问题**：
```python
# Naive fusion 使用 depth proxy
eagle_score = 1.0 / (depth + 1)  # 不可靠！
```

**目标实现**：
```python
# 修改 Eagle3 返回真实 logprob
def topK_genrate(self, hidden_states, input_ids, head):
    ...
    logits = self.lm_head(hidden_states)
    probs = torch.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=self.top_k)
    topk_logprobs = torch.log(topk_probs)  # 新增
    
    return draft_tokens, buffers, topk_logprobs  # 返回 logprob
```

**修改位置**：
- `samd/tree_model/eagle3/eagle3_model.py::topK_genrate()`
- `samd/tree_model/eagle3/eagle3.py::gen_draft()`
- `samd/fusion/naive_fusion.py::parse_eagle_tree()`

**验证**: 在 MT-Bench 上重新评估 naive fusion，预期提升 5-10%

**3.2 P1: 统一 Payoff 校准器**

**目标**: 将 Eagle logprob 和 SAM match_length 统一到 P(accept)

```python
class PayoffCalibrator:
    def __init__(self):
        self.eagle_calibrator = IsotonicRegression()
        self.sam_calibrator = IsotonicRegression()
        self.history = {"eagle": [], "sam": []}
    
    def update(self, verified_nodes):
        """在线更新校准模型"""
        for node in verified_nodes:
            if node.source == "eagle":
                self.history["eagle"].append((node.logprob, node.accepted))
            else:
                self.history["sam"].append((node.match_length, node.accepted))
        
        # 每 N 步重新校准
        if len(self.history["eagle"]) % 100 == 0:
            self.recalibrate()
    
    def predict_accept_prob(self, node) -> float:
        """统一接口：预测接受概率"""
        if node.source == "eagle":
            return self.eagle_calibrator.predict([node.logprob])[0]
        else:
            return self.sam_calibrator.predict([node.match_length])[0]
    
    def compute_payoff(self, node) -> float:
        """期望 payoff = P(accept) * depth"""
        p_accept = self.predict_accept_prob(node)
        return p_accept * node.depth
```

**集成位置**：
- `samd/fusion/calibrator.py` - 新增文件
- `samd/fusion/payoff_fusion.py` - 实现 payoff-aware 融合
- `samd/draft.py` - 添加校准器实例和更新逻辑

**验证**: MT-Bench 上对比 naive + logprob vs payoff-aware，预期额外提升 10-15%

**3.3 P2: Context-aware Fusion Weight**

**目标**: 根据任务特征动态调整 Eagle/SAM 权重

```python
def compute_fusion_weights(context):
    """
    context = {
        "domain": "code" | "chat" | "summarization",
        "entropy": float,  # token 熵
        "sam_quality": float,  # SAM 匹配质量
        "context_length": int,
    }
    """
    w_eagle = 0.5
    w_sam = 0.5
    
    # 代码任务 → SAM 权重高（重复性强）
    if context["domain"] == "code":
        w_sam += 0.2
    
    # 高熵 → Eagle 权重高（需要多样性）
    if context["entropy"] > threshold_high:
        w_eagle += 0.2
    
    # SAM 匹配差 → Eagle 权重高
    if context["sam_quality"] < threshold_low:
        w_eagle += 0.3
    
    return normalize(w_eagle, w_sam)

# 应用到融合
node.weighted_payoff = node.payoff * weight[node.source]
```

**验证**: 跨域测试（MT-Bench + MedQAd），预期额外提升 5-10%

**3.4 实验计划（MT-Bench 为主）**

| Method | Target | Status |
|--------|--------|--------|
| eagle3_only | 143.4 TPS, 5.52 MAT | Baseline |
| sam_sequence_graft | 154.7 TPS, 5.85 MAT (1.044x) | Upper bound |
| naive_fusion | < 130 TPS | ❌ Failed |
| naive + logprob | > 140 TPS | Week 1 |
| + payoff_aware | > 150 TPS | Week 2 |
| + context_aware | **> 157 TPS (1.05x)** ✅ | Week 3 |

**跨域验证（MedQAd）**:
- eagle3_only: 127.5 TPS, 4.97 MAT
- sam_sequence_graft: 138.4 TPS, 5.01 MAT (1.016x)
- **目标**: > 140 TPS (1.02x)

**3.5 实施时间线**

- **Week 1**: 提取 Eagle3 logprob + MT-Bench 验证
- **Week 2**: Payoff 校准器 + MT-Bench 完整测试
- **Week 3**: Context-aware + 跨域验证（MedQAd）
- **Week 4**: 论文实验 + 初稿撰写
```python
class PayoffCalibrator:
    def __init__(self):
        self.eagle_history = []
        self.sam_history = []
        self.calibrator = IsotonicRegression()
    
    def update(self, verified_nodes):
        # 记录 (score, accepted) pairs
    
    def calibrate(self):
        # 每 N 步重新训练
```

**3.3 实现集成**
- 添加 `FusionMode.PAYOFF_AWARE`
- 修改 `gen_candidates()` 调用 `fuse_dual_drafts()`
- 集成 calibrator 到 decode loop

### Phase 4: Tree-Aware Insertion

**4.1 插入位置策略**
```python
def identify_insertion_points(eagle_tree, context):
    points = []
    
    # 策略1: 根节点后备
    points.append(root_fallback())
    
    # 策略2: 高置信度路径 -> sibling extension
    for node in high_confidence_leaves:
        points.append(sibling_insertion(node))
    
    # 策略3: 低置信度路径 -> fallback path
    for node in low_confidence_nodes:
        points.append(fallback_insertion(node))
    
    return top_k(points, by=priority)
```

**4.2 SAM 分支生成**
- 为每个 insertion point 生成 SAM 分支
- 长度自适应：高优先级 -> 更长分支
- 集成到 trie 构建

**4.3 实现优化**
- 添加 `FusionMode.TREE_AWARE`
- 并行化 SAM 分支生成
- GPU-resident adjacency matrix（参考 Graft）

### Phase 5: 实验验证

**5.1 对比实验**
| Method | Description |
|--------|-------------|
| Eagle3-only | Baseline |
| SAM-only | Baseline |
| HVD-style | 选择模式：if sam_quality > θ then SAM else Eagle |
| Graft-style | 串行模式：prune Eagle → fill with SAM |
| Naive fusion | max confidence + fixed truncate |
| Payoff-aware | 我们的方法（无 tree-aware） |
| Full method | 我们的方法（完整版） |

**5.2 评估指标**
```python
metrics = {
    # 核心指标
    "speedup": tokens / (draft_time + verify_time),
    "MAT": mean_accepted_per_step,
    "verified_per_accepted": efficiency,
    
    # 来源分析
    "eagle_contribution": accepted_from_eagle / total,
    "sam_contribution": accepted_from_sam / total,
    "agreement_rate": both_proposed_accepted / both_proposed,
    
    # 校准质量
    "calibration_ECE": expected_calibration_error,
    "fusion_weight_trace": (w_eagle, w_sam) over time,
}
```

**5.3 消融实验**
1. Fusion weight: 固定 0.5/0.5 vs context-aware
2. Insertion: root-only vs tree-aware
3. Budget: 50/50 split vs dynamic allocation
4. Calibration: no calibration vs online calibration

**5.4 场景分析**
- Code generation (HumanEval, RepoBench)
- Math reasoning (GSM8K)
- Summarization (CNN/DM, GovReport)
- Long context (LongBench suite)
- High/low temperature sampling

## 交付物

### 研究文档
- [x] PRD (本文档)
- [ ] 理论推导文档
- [ ] 实验设计文档
- [ ] 结果分析报告

### 代码实现
- [ ] `samd/fusion/` 模块
  - [ ] `trie.py`: FusionTrie 实现
  - [ ] `calibrator.py`: PayoffCalibrator 实现
  - [ ] `context.py`: Context feature extraction
  - [ ] `policy.py`: Fusion weight computation
  - [ ] `insertion.py`: Tree-aware insertion strategy
  - [ ] `fusion.py`: 主融合逻辑
- [ ] 集成到 `samd/utils.py::gen_candidates()`
- [ ] 实验脚本 `evaluation/run_fusion_experiments.py`

### 论文素材
- [ ] Algorithm pseudocode
- [ ] 架构图（对比 Graft/HVD）
- [ ] 实验结果表格
- [ ] 消融实验图表
- [ ] Case study 示例

## 时间规划

- **Week 1**: Phase 1 理论框架 + Phase 2 Naive baseline
- **Week 2**: Phase 3 Payoff-aware fusion + 初步实验
- **Week 3**: Phase 4 Tree-aware insertion + 完整实验
- **Week 4**: Phase 5 对比实验 + 消融实验 + 论文撰写

## 风险评估

**技术风险**：
1. 校准可能需要大量数据 → 使用 warm-up + online learning
2. Tree-aware insertion 复杂度高 → 限制 insertion points 数量
3. 性能开销可能抵消收益 → GPU-resident 优化 + 并行化

**创新性风险**：
1. 与 Graft/HVD 区分度不够 → 强调并行 vs 串行/选择的本质差异
2. 实验结果不显著 → 聚焦特定场景（code/long-context）证明优势
3. 理论深度不够 → 补充 payoff estimation 的理论分析

## 成功标准

**最低标准**（可发表）：
- Naive fusion 在至少 2 个场景下超过 HVD 和 Graft
- 消融实验证明各组件有效

**期望标准**（强接收）：
- Payoff-aware fusion 在多数场景下 SOTA
- 理论框架清晰，校准有效
- Tree-aware 在 code/long-context 有显著提升

**理想标准**（高引用潜力）：
- 统一的 multi-source draft fusion 框架
- 适用于 Eagle/SAM 之外的其他 drafter 组合
- 开源实现 + 易集成到现有系统
