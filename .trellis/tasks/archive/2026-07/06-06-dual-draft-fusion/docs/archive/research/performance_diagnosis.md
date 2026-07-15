# Naive Fusion 性能诊断与改进方案

## 实验结果回顾

```
Method          TPS     MAT    Steps   Relative
Eagle3-only    60.89   7.009   53.20   1.00x (baseline)
SAM-only       21.71   1.493  249.70   0.36x
naive_fusion   49.62   5.936   62.43   0.81x ⚠️
```

**核心问题**：naive_fusion 比 Eagle3-only 慢 19%

**观察**：
- SAM contribution ~20-30/60 候选节点（33-50%）
- both_contribution 很低（双引擎一致性差）
- 动态 SAM（基于 prompt），无静态 SAM

---

## 问题 1：为什么 naive fusion 比 Eagle3-only 慢？

### 假设 A：SAM 候选质量低，拉低整体接受率

**证据**：
- MAT: 5.936 < 7.009（降低 15%）
- Steps: 62.43 > 53.20（增加 17%）
- SAM 占 33-50% 候选，但 both_contribution 低

**推理**：
```
如果 SAM 候选被频繁拒绝：
→ 浪费验证 token 预算（60个中的20-30个）
→ 真正有用的 Eagle 候选被挤掉
→ MAT 下降，Steps 增加
→ TPS 下降
```

**如何验证**：
- 统计 SAM-only 候选的接受率 vs Eagle-only 候选的接受率
- 对比融合树中 SAM 节点的实际被接受比例

### 假设 B：分数归一化错误，误杀高质量 Eagle 候选

**证据**：
- Eagle score: `1.0 / (depth + 1)` - 深度代理
- SAM score: `match_length - position`
- 分别归一化后排序

**问题**：
```python
# Eagle 候选分数分布
Eagle depth=1: score=1.0  → 归一化后 ~1.0
Eagle depth=2: score=0.5  → 归一化后 ~0.5
Eagle depth=4: score=0.2  → 归一化后 ~0.2

# SAM 候选分数分布（假设 match_length=10）
SAM pos=1: score=9  → 归一化后 ~1.0
SAM pos=5: score=5  → 归一化后 ~0.56

# 融合排序后可能：
# SAM pos=1 (0.9) > Eagle depth=1 (0.8) ❌ 错误排序
```

**根本问题**：
- Eagle depth proxy 不反映真实接受概率
- SAM match_length 在动态 SAM 下意义不明确
- 两者归一化到同一尺度**假设它们可比**，但实际不可比

**如何验证**：
- 记录每个节点的分数和最终是否被接受
- 计算 Eagle/SAM 分数与接受率的相关性

### 假设 C：树结构次优，验证开销增加

**证据**：
- SAM 是线性序列，转换成树结构
- 融合后的树可能不如纯 Eagle 树紧凑

**问题**：
```
Eagle3 树是训练优化过的（topK_genrate）
→ 树宽度/深度平衡，attention 计算高效

SAM 线性链 + Eagle 树融合
→ 可能产生"瘦高"或"矮胖"的树
→ attention mask 更复杂
→ 验证时间增加
```

**如何验证**：
- Profile 验证阶段的时间
- 对比 Eagle-only 和 naive_fusion 的单步验证时间

### 假设 D：动态 SAM 匹配质量差

**证据**：
- 当前用动态 SAM（基于 prompt 构建）
- 没有静态 SAM（避免数据泄露）

**问题**：
```
动态 SAM 仅从 prompt 中提取 token 转移
→ 匹配模式少（HumanEval prompt 通常 <100 tokens）
→ 生成任务中 prompt 和生成内容可能差异大
→ SAM 匹配长度短，候选质量低
```

**对比**：
```
静态 SAM（训练集构建）:
→ 数万个 token 转移模式
→ 匹配长度长，候选质量高

动态 SAM（单个 prompt）:
→ 几十个 token 转移模式
→ 匹配长度短，候选质量低
```

**如何验证**：
- 统计动态 SAM 的平均匹配长度
- 对比 Eagle-only 在相同位置的接受率

---

## 问题 2：动态 SAM 的问题在哪？如何验证？

### 动态 SAM 的局限性

**1. 匹配模式稀疏**
```python
# HumanEval 典型 prompt
"""
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    \"\"\"
"""

# 动态 SAM 能学到的模式：
# "from" → "typing"
# "def" → "has_close_elements"
# ...只有 prompt 中出现的转移

# 但生成时需要的模式：
# "for" → "i"
# "return" → "True"
# "if" → "abs"
# ...这些在 prompt 中可能没有
```

**2. 域不匹配**
```
Prompt: 函数签名 + docstring（自然语言）
Generation: 函数体（代码逻辑）

动态 SAM 从 prompt 学到的是"声明"模式
但生成需要的是"逻辑"模式
→ 域 mismatch，匹配失败
```

**3. 上下文窗口限制**
```
动态 SAM 只能看到当前 prompt + 已生成内容
→ 无法利用"其他类似问题"的模式
→ 泛化能力弱
```

### 验证方案

**方案 1：统计 SAM 匹配质量**
```python
# 在 gen_candidates() 中记录
sam_stats = {
    "avg_match_length": [],
    "max_match_length": [],
    "sam_contribution": [],
    "sam_accept_rate": [],
}

# 对比 Eagle-only 的接受率
eagle_accept_rate = accepted_eagle / total_eagle
sam_accept_rate = accepted_sam / total_sam

# 如果 sam_accept_rate << eagle_accept_rate
# → 证明 SAM 候选质量低
```

**方案 2：对比不同 SAM 来源**
```bash
# A. 动态 SAM（当前）
python eval.py --fusion_mode naive  # 无 --sam_path

# B. 跨域静态 SAM（训练集不重叠）
# 例如：用 MBPP 训练 SAM，在 HumanEval 测试
python tools/build_sam.py --dataset mbpp --output sam_mbpp.pkl
python eval.py --fusion_mode naive --sam_path sam_mbpp.pkl

# 如果 B 显著优于 A
# → 证明问题在动态 SAM 匹配质量
```

**方案 3：Case study 分析**
```python
# 选择 TPS 最低的 10 个样本
# 可视化融合树：
# - Eagle 候选在哪里
# - SAM 候选在哪里
# - 哪些被接受，哪些被拒绝

# 看是否有模式：
# - SAM 总是在深层（深度 > 5）
# - SAM 候选都是低频 token
# - SAM 匹配长度都很短（< 3）
```

---

## 问题 3：如何在不泄露数据的前提下使用 SAM？

### 方案对比

| 方案 | 描述 | 优点 | 缺点 | 数据泄露风险 |
|------|------|------|------|-------------|
| **A. 动态 SAM** | 从 prompt 构建 | 无泄露 | 质量差 | ✅ 无风险 |
| **B. 跨域静态 SAM** | 用其他数据集训练 | 质量好 | 域不匹配 | ✅ 无风险 |
| **C. Leave-one-out** | HumanEval 分 fold | 质量最好 | 复杂 | ⚠️ 轻微泄露 |
| **D. 通用代码 SAM** | 用大规模代码训练 | 质量好 | 需要资源 | ✅ 无风险 |

### 推荐方案

**Phase 2（当前）：方案 A + 方案 B**
```bash
# 1. 保留动态 SAM（证明 naive fusion 概念可行）
python eval.py --fusion_mode naive  # baseline

# 2. 添加跨域静态 SAM（证明 SAM 质量影响）
python tools/build_sam.py --dataset mbpp --output sam_mbpp.pkl
python tools/build_sam.py --dataset apps --output sam_apps.pkl

python eval.py --fusion_mode naive --sam_path sam_mbpp.pkl
python eval.py --fusion_mode naive --sam_path sam_apps.pkl

# 论文中报告：
# - 动态 SAM：概念验证
# - MBPP SAM：跨域泛化能力
# - 讨论：静态 SAM 在同域会更好（但不能实验）
```

**Phase 3（Payoff-aware）：方案 D**
```bash
# 训练通用代码 SAM
# 数据源：GitHub 公开代码（排除 HumanEval/MBPP 仓库）
# 大小：1M+ token 转移

# 优点：
# - 覆盖广泛的代码模式
# - 无数据泄露
# - 可以开源给社区使用

# 实现：
python tools/build_sam.py \
  --dataset the-stack \
  --languages python \
  --exclude humaneval,mbpp \
  --output sam_general_code.pkl
```

### Leave-one-out 方案（可选，但需谨慎）

```python
# HumanEval 有 164 题
# 分成 10 fold，每次用 9 fold 训练 SAM，在 1 fold 测试

# 优点：测试集与训练集同域，质量最好
# 缺点：
# 1. 理论上有"见过相似代码"的风险
# 2. 需要跑 10 次实验
# 3. 可能被 reviewer 质疑

# 建议：仅作为 upper bound 分析，不作为主要结果
```

---

## 问题 4：分数融合的根本问题是什么？

### 当前方案的缺陷

```python
# 当前实现（samd/fusion/naive_fusion.py）
def sort_by_score(nodes):
    # 分别归一化
    eagle_scores = normalize([n.score for n in eagle_nodes])
    sam_scores = normalize([n.score for n in sam_nodes])

    # 合并排序
    all_scores = eagle_scores + sam_scores
    return sorted(nodes, key=lambda n: n.normalized_score, reverse=True)
```

**问题 1：分数语义不同**
```
Eagle score (depth proxy):
- score = 1.0 / (depth + 1)
- 语义：位置偏好（浅层优先）
- 不反映：接受概率

SAM score (match_length - position):
- score = match_length - position
- 语义：匹配长度 - 在序列中的位置
- 不反映：接受概率
```

**问题 2：归一化假设错误**
```python
# 归一化假设：分数在源内是可比的
eagle_normalized = (score - min_eagle) / (max_eagle - min_eagle)
sam_normalized = (score - min_sam) / (max_sam - min_sam)

# 但实际：
# - Eagle depth=1 (norm=1.0) 可能接受率 80%
# - SAM match_len=10 (norm=1.0) 可能接受率 30%
# → 归一化后的 1.0 意义完全不同！
```

**问题 3：忽略上下文**
```
同样的 Eagle 候选，在不同场景下接受率不同：
- Code 任务：Eagle 接受率高（重复性强）
- Creative 任务：Eagle 接受率低（需要多样性）

同样的 SAM 候选，在不同场景下接受率不同：
- 长上下文：SAM 接受率高（模式丰富）
- 短上下文：SAM 接受率低（模式稀疏）

→ 固定的归一化无法适应不同场景
```

### 根本问题：缺少统一的目标函数

**当前**：
```
目标：选择"分数高"的候选
问题：分数 ≠ 价值
```

**应该**：
```
目标：最大化 E[accepted_tokens]
需要：P(accept | candidate, context) 的准确估计
```

### 正确的融合目标

```python
# 目标：最大化期望接受 token 数
def fuse_with_payoff(eagle_nodes, sam_nodes, context):
    for node in all_nodes:
        # 统一目标：期望接受 token 数
        node.payoff = estimate_payoff(node, context)
        # payoff = P(accept) × depth

    # 前缀闭包约束下选择高 payoff 节点
    selected = select_with_prefix_closure(all_nodes, budget=60)
    return selected

def estimate_payoff(node, context):
    # 关键：估计 P(accept)
    if node.source == "eagle":
        p_accept = calibrate_eagle(node.logprob, context)
    else:  # SAM
        p_accept = calibrate_sam(node.match_length, context)

    return p_accept * node.depth
```

---

## 问题 5：Phase 3 payoff-aware 应该优先解决哪个问题？

### 优先级排序

#### **P0：提取 Eagle3 真实 logprob**

**为什么最优先**：
- 当前 depth proxy 完全不可靠
- Eagle3 本身就有 logprob，只是没返回
- 修改成本低，收益大

**实现**：
```python
# 修改 samd/tree_model/eagle3/eagle3_model.py
def topK_genrate(self, hidden_states, input_ids, head):
    ...
    # 现有代码已经计算了 logits
    logits = self.lm_head(hidden_states)
    probs = torch.softmax(logits, dim=-1)

    # 添加：返回 topk 的 logprob
    topk_probs, topk_indices = torch.topk(probs, k=self.top_k)
    topk_logprobs = torch.log(topk_probs)

    return draft_tokens, buffers, topk_logprobs  # 新增返回

# 修改 samd/tree_model/eagle3/eagle3.py
def gen_draft(self, start_token):
    draft_tokens, buffers, logprobs = self.model.topK_genrate(...)
    return draft_tokens, buffers, logprobs  # 传递
```

**预期收益**：
- 替换 depth proxy 为真实 logprob
- 融合排序更准确
- 预计提升 5-10% TPS

#### **P1：实现基础 Payoff 校准**

**目标**：
```python
# 将 Eagle logprob 和 SAM match_length 统一到 P(accept)
def calibrate_eagle_logprob(logprob, history):
    # Isotonic regression: logprob → P(accept)
    # 使用验证历史：(logprob, was_accepted) pairs
    return calibrator_eagle.predict(logprob)

def calibrate_sam_match_length(match_length, history):
    # Isotonic regression: match_length → P(accept)
    return calibrator_sam.predict(match_length)
```

**实现**：
```python
# samd/fusion/calibrator.py
from sklearn.isotonic import IsotonicRegression

class PayoffCalibrator:
    def __init__(self):
        self.eagle_calibrator = IsotonicRegression()
        self.sam_calibrator = IsotonicRegression()
        self.history_eagle = []
        self.history_sam = []

    def update(self, verified_nodes):
        for node in verified_nodes:
            if node.source == "eagle":
                self.history_eagle.append((node.logprob, node.accepted))
            else:
                self.history_sam.append((node.match_length, node.accepted))

        # 每 N 步重新校准
        if len(self.history_eagle) % 100 == 0:
            self.recalibrate()

    def recalibrate(self):
        if len(self.history_eagle) > 50:
            X = [x[0] for x in self.history_eagle]
            y = [x[1] for x in self.history_eagle]
            self.eagle_calibrator.fit(X, y)

        if len(self.history_sam) > 50:
            X = [x[0] for x in self.history_sam]
            y = [x[1] for x in self.history_sam]
            self.sam_calibrator.fit(X, y)

    def predict_accept_prob(self, node):
        if node.source == "eagle":
            return self.eagle_calibrator.predict([node.logprob])[0]
        else:
            return self.sam_calibrator.predict([node.match_length])[0]
```

**预期收益**：
- 统一融合目标（P(accept)）
- 自适应不同样本
- 预计提升 10-15% TPS

#### **P2：添加 Context-aware fusion weight**

**目标**：
```python
# 根据任务类型调整 Eagle/SAM 权重
def compute_fusion_weights(context):
    w_eagle = 0.5
    w_sam = 0.5

    # 代码任务 → SAM 权重高
    if detect_code_task(context):
        w_sam += 0.2

    # 高熵 → Eagle 权重高
    if compute_entropy(context) > threshold:
        w_eagle += 0.2

    return normalize(w_eagle, w_sam)

# 融合时应用权重
node.weighted_payoff = node.payoff * weight[node.source]
```

**预期收益**：
- 场景自适应
- Code 任务提升更多
- 预计额外提升 5-10% TPS

#### **P3：Tree-aware insertion（可选）**

**说明**：
- 只在 P0-P2 都实现且仍有提升空间时考虑
- 复杂度高，收益不确定

---

## 诊断结论与行动计划

### 主要问题诊断

**根本原因**：分数融合不准确
1. ✅ **已确认**：Eagle depth proxy 不可靠
2. ⚠️ **高度怀疑**：动态 SAM 质量差
3. ⚠️ **待验证**：SAM 候选被频繁拒绝，浪费预算

**次要原因**：
4. ⚠️ **可能存在**：树结构次优，验证开销增加

### 立即行动（本周）

**Step 1：数据收集与分析**
```bash
# 修改代码输出详细统计
# 添加到 samd/draft.py::record_naive_fusion()
stats = {
    "eagle_nodes": len(eagle),
    "sam_nodes": len(sam),
    "eagle_accepted": sum(n.accepted for n in eagle),
    "sam_accepted": sum(n.accepted for n in sam),
    "eagle_accept_rate": eagle_accepted / eagle_nodes,
    "sam_accept_rate": sam_accepted / sam_nodes,
    "avg_sam_match_length": mean([n.match_length for n in sam]),
}

# 重新跑评估，收集每个样本的统计
python eval.py --fusion_mode naive > fusion_stats.jsonl
```

**Step 2：对比跨域静态 SAM**
```bash
# 用 MBPP 训练 SAM
python tools/build_sam.py --dataset mbpp --output sam_mbpp.pkl

# 在 HumanEval 测试
python eval.py --fusion_mode naive --sam_path sam_mbpp.pkl

# 如果显著优于动态 SAM → 证明问题在 SAM 质量
# 如果仍然慢 → 证明问题在分数融合
```

**Step 3：提取 Eagle3 logprob**
```python
# 修改 eagle3_model.py 返回 logprob
# 修改 naive_fusion.py 使用真实 logprob
# 重新评估

# 预期：至少达到 0.9x Eagle3-only
```

### Phase 3 实现顺序（下周）

```
Week 1: P0 Eagle3 logprob 提取
Week 2: P1 基础 Payoff 校准
Week 3: P2 Context-aware weight
Week 4: 完整实验 + 论文撰写
```

---

## 论文策略调整

**原计划**：
- Naive fusion 是 baseline，应该接近 Eagle-only
- Payoff-aware 是主要贡献

**新策略（如果 naive 持续慢）**：
- **Naive fusion 定位为"negative baseline"**
- 强调：简单融合不够，需要智能校准
- 故事线：
  1. Motivation: Eagle + SAM 应该互补
  2. Naive baseline: 简单融合反而更慢（分析原因）
  3. Our method: Payoff-aware 解决融合问题
  4. Results: 超过 Eagle-only 和 Graft/HVD

**优势**：
- 有 negative result 更真实
- 突出 payoff-aware 的必要性
- 避免 reviewer 质疑"为什么不直接 union"
