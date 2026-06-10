# Phase 3.2: Payoff-Aware Fusion 设计方案

## 问题分析

### Profile 结果

```
验证逻辑 (eval_posterior): 27.4% ← 主要开销
Fusion 代码: < 1%
```

### 根本原因

**1. 候选数过多**：
```
Pure Eagle3:        ~42 nodes
sam_sequence_graft: ~50 nodes (controlled)
naive_fusion:       ~60 nodes (43% more!) ❌
```

**2. 验证开销 ∝ 候选数**：
```
验证时间增加 → 整体 TPS 下降
```

**3. 分数融合不准确**：
```
选择了低质量候选 → 浪费验证时间 → 接受率低
```

---

## 解决方案：Payoff-Aware Fusion

### 核心思想

**Payoff = P(accept) × depth**

- 不是简单的分数归一化
- 估计每个候选的期望收益
- 选择高 payoff 候选，控制总预算

### 设计目标

1. **减少候选数**：60 → 50（减少 17% 验证开销）
2. **提高候选质量**：选择高 payoff 节点
3. **超过 sam_sequence_graft**：> 56.2 TPS

---

## Payoff Estimator 设计

### 简单启发式（Phase 3.2.1）

**Eagle: logprob → P(accept)**

```python
def estimate_eagle_accept_prob(logprob):
    """
    将 Eagle logprob 映射到接受概率
    
    Args:
        logprob: float, 范围 [-20, 0]
    
    Returns:
        p_accept: float, 范围 [0, 1]
    """
    # 启发式 1：Sigmoid 映射
    # logprob = 0 → p ≈ 0.73
    # logprob = -5 → p ≈ 0.27
    # logprob = -10 → p ≈ 0.05
    
    # 调整参数使分布合理
    # 观察：大部分 logprob 在 [-10, -2] 范围
    # 目标：让这个范围映射到 [0.1, 0.9]
    
    k = 0.5  # 调节陡度
    logprob_shifted = logprob + 5  # 平移中心
    p_accept = 1 / (1 + math.exp(-k * logprob_shifted))
    
    return p_accept
```

**SAM: match_length → P(accept)**

```python
def estimate_sam_accept_prob(match_length):
    """
    将 SAM match_length 映射到接受概率
    
    Args:
        match_length: int, 范围 [0, 40]
    
    Returns:
        p_accept: float, 范围 [0, 1]
    """
    # 启发式 2：线性映射 + 饱和
    # match_length = 0 → p = 0
    # match_length = 10 → p = 0.5
    # match_length ≥ 20 → p = 1.0
    
    # 观察：match_length > 10 的候选通常质量好
    # 目标：match_length [5, 15] → p [0.25, 0.75]
    
    max_length = 20
    p_accept = min(match_length / max_length, 1.0)
    
    return p_accept
```

**Payoff 计算**：

```python
def compute_payoff(node):
    """
    计算候选的期望 payoff
    
    Payoff = P(accept) × depth
    """
    if node.source == "eagle":
        p_accept = estimate_eagle_accept_prob(node.score)
    else:  # SAM
        p_accept = estimate_sam_accept_prob(node.score)
    
    payoff = p_accept * node.depth
    return payoff
```

---

## Fusion 算法修改

### 当前 naive_fusion

```python
def fuse_eagle_sam_naive(eagle_tree, sam_candidates, ...):
    # 1. 解析
    eagle_nodes = parse_eagle_tree(eagle_tree)  # 42 nodes
    sam_nodes = parse_sam_sequence(sam_candidates)  # 18 nodes
    
    # 2. 分数归一化
    eagle_scores_norm = normalize([n.score for n in eagle_nodes])
    sam_scores_norm = normalize([n.score for n in sam_nodes])
    
    # 3. 合并去重
    merged_nodes = merge_and_dedup(eagle_nodes, sam_nodes)  # 60 nodes
    
    # 4. 按归一化分数排序
    merged_nodes.sort(key=lambda n: n.normalized_score, reverse=True)
    
    # 5. 截断到预算
    selected_nodes = truncate_with_ancestors(merged_nodes, max_tokens=60)
    
    return selected_nodes
```

### 改进：payoff_fusion

```python
def fuse_eagle_sam_payoff(eagle_tree, sam_candidates, config):
    # 1. 解析
    eagle_nodes = parse_eagle_tree(eagle_tree, eagle_logprobs)
    sam_nodes = parse_sam_sequence(sam_candidates, sam_match_length)
    
    # 2. 计算 payoff（不需要归一化！）
    for node in eagle_nodes:
        node.payoff = compute_payoff(node)
    
    for node in sam_nodes:
        node.payoff = compute_payoff(node)
    
    # 3. 合并去重（保留高 payoff）
    merged_nodes = merge_and_dedup_by_payoff(eagle_nodes, sam_nodes)
    
    # 4. 按 payoff 排序
    merged_nodes.sort(key=lambda n: n.payoff, reverse=True)
    
    # 5. 贪心选择（控制预算）
    max_tokens = config.get("max_draft_tokens", 50)  # 减少到 50
    selected_nodes = select_with_budget(merged_nodes, max_tokens)
    
    return selected_nodes


def select_with_budget(nodes, budget):
    """
    贪心选择高 payoff 节点，确保前缀闭包
    """
    selected_keys = set()
    selected_nodes = []
    
    for node in nodes:  # 已按 payoff 排序
        # 收集祖先
        ancestors = collect_ancestors(node)
        needed_keys = {get_key(a) for a in ancestors} | {get_key(node)}
        
        # 检查预算
        if len(selected_keys | needed_keys) <= budget:
            selected_keys.update(needed_keys)
            selected_nodes.append(node)
    
    return selected_nodes
```

---

## 实施计划

### Step 1: 实现 Payoff Estimator

**创建文件**：`samd/fusion/payoff_estimator.py`

```python
import math

class PayoffEstimator:
    def __init__(self, config=None):
        self.config = config or {}
        # Eagle sigmoid 参数
        self.eagle_k = self.config.get("eagle_k", 0.5)
        self.eagle_shift = self.config.get("eagle_shift", 5.0)
        # SAM 线性参数
        self.sam_max_length = self.config.get("sam_max_length", 20)
    
    def estimate_eagle_payoff(self, logprob, depth):
        logprob_shifted = logprob + self.eagle_shift
        p_accept = 1 / (1 + math.exp(-self.eagle_k * logprob_shifted))
        return p_accept * depth
    
    def estimate_sam_payoff(self, match_length, depth):
        p_accept = min(match_length / self.sam_max_length, 1.0)
        return p_accept * depth
```

### Step 2: 修改 Fusion 模块

**修改文件**：`samd/fusion/payoff_fusion.py`

1. 复制 `naive_fusion.py` 为 `payoff_fusion.py`
2. 修改 `fuse_eagle_sam_payoff()` 使用 payoff
3. 减少 `max_draft_tokens` 到 50

### Step 3: 集成到主流程

**修改文件**：`samd/utils.py`

```python
if samd_config.fusion_mode == "payoff_aware":
    # 创建 payoff estimator
    from samd.fusion.payoff_estimator import PayoffEstimator
    estimator = PayoffEstimator(samd_config.payoff_config)
    
    # 融合
    fused_tree = fuse_eagle_sam_payoff(
        eagle_tree=eagle_tree,
        sam_candidates=sam_candidates,
        config=samd_config.fusion_config,
        estimator=estimator,
    )
    # ...
```

### Step 4: 配置扩展

**修改文件**：`samd/samd_config.py`

```python
@dataclass
class SamdConfig:
    # ... 现有字段
    
    # Fusion 配置
    fusion_mode: Literal["none", "naive", "payoff_aware"] = "none"
    fusion_max_draft_tokens: int = 50  # 从 60 减少到 50
    
    # Payoff 配置
    payoff_config: Dict = field(default_factory=lambda: {
        "eagle_k": 0.5,
        "eagle_shift": 5.0,
        "sam_max_length": 20,
    })
```

---

## 验证计划

### Experiment: Payoff-Aware vs Naive

```bash
# Baseline
pure_eagle3: 55.917 TPS

# Current
naive_logprob: 53.539 TPS (0.957x)

# Target (payoff-aware)
payoff_aware: > 56.2 TPS (> 1.001x) ✅
```

**测试命令**：

```bash
# MT-Bench 完整测试
python evaluation/inference_samd.py \
  --fusion_mode payoff_aware \
  --fusion_max_draft_tokens 50 \
  --bench_name mt_bench \
  --answer_file mt_bench_payoff_aware.jsonl
```

**预期改进**：

1. **候选数减少**：60 → 50（-17%）
2. **候选质量提升**：高 payoff 优先
3. **验证效率提升**：减少低价值验证
4. **TPS 提升**：53.5 → 56.5（+5.6%）

---

## 预期结果

### 性能目标

| Method | MAT | TPS | vs Eagle3 | 状态 |
|--------|-----|-----|-----------|------|
| pure_eagle3 | 5.52 | 55.92 | 1.000x | Baseline |
| sam_sequence_graft | 5.85 | 56.21 | 1.005x | Current SOTA |
| naive_logprob | 6.31 | 53.54 | 0.957x | Phase 3.1 |
| **payoff_aware** | **> 5.6** | **> 56.5** | **> 1.01x** | **Phase 3.2 目标** |

### 成功标准

**✅ 成功**：
- TPS > 56.2 (超过 sam_sequence_graft)
- MAT 不显著下降（< 5%）
- 候选数控制在 50 以内

**⚠️ 部分成功**：
- TPS 55-56 (接近但未超过)
- 需要调整 payoff 参数

**❌ 失败**：
- TPS < 55 (未改善)
- 需要重新设计 payoff 估计

---

## 参数调优

### 如果效果不理想

**调整 Eagle sigmoid 参数**：

```python
# 更陡峭（更区分高低 logprob）
eagle_k = 1.0  # vs 0.5

# 不同平移（调整中心）
eagle_shift = 3.0  # vs 5.0
```

**调整 SAM 线性参数**：

```python
# 更宽松（低 match_length 也有机会）
sam_max_length = 30  # vs 20

# 更严格
sam_max_length = 15
```

**调整预算**：

```python
# 更激进（更少候选）
max_draft_tokens = 45  # vs 50

# 更保守
max_draft_tokens = 55
```

---

## 时间估算

- **Step 1-2**: 实现 Payoff Estimator + Fusion（2-3 小时）
- **Step 3-4**: 集成 + 配置（1 小时）
- **验证**: MT-Bench 完整测试（1 小时）
- **调优**: 如果需要（1-2 小时）

**总计**：1 天可以完成完整实现和初步验证

---

## 下一步

1. **今天/明天**：实现 Phase 3.2
2. **验证结果**：MT-Bench 测试
3. **如果成功**：Phase 3.3 (context-aware，可选)
4. **论文实验**：完整对比实验

---

## Codex 执行（可选）

```bash
codex exec "实现 Phase 3.2: Payoff-Aware Fusion。

参考规格：.trellis/tasks/06-06-dual-draft-fusion/docs/implementation/phase3.2_payoff_aware.md

任务：
1. 创建 samd/fusion/payoff_estimator.py
2. 创建 samd/fusion/payoff_fusion.py
3. 修改 samd/utils.py 添加 fusion_mode=payoff_aware
4. 修改 samd/samd_config.py 添加配置

要求：
- Eagle logprob → P(accept) via sigmoid
- SAM match_length → P(accept) via linear
- 控制候选预算到 50
- 按 payoff 排序选择

验证：编译检查 + 生成测试命令" \
  --cd /path/to/SAM-Decoding \
  --sandbox workspace-write
```
