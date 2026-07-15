# Phase 3.1 Profiling - 执行指南

## 目标

Profile naive_logprob (fusion_mode=naive, threshold=5) 找到性能瓶颈，决定下一步：
- **有明显瓶颈** → 代码优化
- **无明显瓶颈** → 直接进入 Phase 3.2（Payoff-Aware）

---

## 远端执行

### Step 1: 同步代码

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 拉取最新代码（包含 profile 工具）
git fetch origin feature/eagle3-tail-sidecar
git pull origin feature/eagle3-tail-sidecar

# 或从本地同步
# - scripts/profile_naive_fusion.py
# - scripts/run_profile.sh
```

### Step 2: 运行 Profiler

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 设置环境变量（如果 .env 不存在）
export MODEL_PATH="/teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct"
export TREE_MODEL_PATH="/teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B"

# 运行 profiler（10 个样本）
cd scripts
bash run_profile.sh 2 12

# 或直接运行 Python 脚本
cd ..
PYTHONPATH=. python scripts/profile_naive_fusion.py \
  --question_begin 2 \
  --question_end 12 \
  --output profile_naive_th5.txt
```

**预计时间**：5-10 分钟

### Step 3: 查看结果

```bash
# 查看完整报告
cat profile_naive_th5.txt

# 或查看关键部分
grep -A 20 "=== Top 20" profile_naive_th5.txt
grep -A 10 "Fusion-specific" profile_naive_th5.txt
grep -A 5 "Recommendation" profile_naive_th5.txt
```

---

## 输出解读

### 报告格式

```
=== Profiling Results ===
Total runtime: 45.2 seconds
Samples analyzed: 10 questions

=== Top 20 Functions by Cumulative Time ===
  1. generate (samd_model.py): 15.2s (33.6%)
  2. forward (llama.py): 8.3s (18.4%)
  3. fuse_eagle_sam_naive: 2.1s (4.6%)  ← 关注这个
  ...

=== Fusion-Specific Breakdown ===
  - fuse_eagle_sam_naive: 4.6%
  - parse_eagle_tree: 1.2%
  - normalize_scores: 0.8%
  ...

=== Recommendation ===
[X] No obvious bottleneck in fusion code (all < 5%)
[ ] Proceed to Phase 3.2 (Payoff-Aware optimization)

OR

[ ] Code optimization needed
[X] Function `fuse_eagle_sam_naive` takes 8.2% time
    → Investigate and optimize this function
```

### 判断标准

**场景 A：无明显瓶颈**
```
- 所有 fusion-specific 函数 < 5%
- fuse_eagle_sam_naive < 5%
→ 代码实现合理，直接进入 Phase 3.2
```

**场景 B：有瓶颈**
```
- 某个函数 > 5%（如 fuse_eagle_sam_naive: 8%）
→ 需要优化该函数
→ 优化后重新 profile
```

---

## 下一步决策

### 如果是场景 A（推荐）

**结论**：代码实现合理，性能损失来自融合算法本身

**下一步**：直接进入 Phase 3.2 - Payoff-Aware Fusion

**理由**：
1. 当前 naive fusion 使用简单的分数归一化
2. Eagle logprob 和 SAM match_length 难以直接比较
3. 没有考虑候选的期望收益（payoff）
4. **Payoff-aware 才是我们的创新点**

**Phase 3.2 目标**：
```python
# 当前：简单分数融合
score = normalize([eagle_logprob, sam_match_length])

# Phase 3.2：基于 payoff 选择
payoff = P(accept) * depth  # 期望接受 token 数
```

**预期提升**：5-10%，达到 56+ TPS

### 如果是场景 B

**需要优化的常见瓶颈**：

1. **parse_eagle_tree**：
   - 树结构解析
   - 可能的优化：缓存、预计算

2. **normalize_scores**：
   - 分数归一化
   - 可能的优化：向量化操作

3. **merge_and_dedup**：
   - 去重逻辑
   - 可能的优化：哈希表、集合操作

4. **truncate_with_ancestors**：
   - 前缀闭包构建
   - 可能的优化：BFS 优化、早停

**优化后**：重新 profile 验证

---

## Phase 3.2 预览

### 当前性能

```
pure_eagle3:        55.917 TPS (baseline)
sam_sequence_graft: 56.211 TPS (1.005x) ← 目标超越
naive_logprob:      53.539 TPS (0.957x) ← 当前
```

### Phase 3.2 方案

**核心改进**：统一 Payoff 估计

```python
class PayoffEstimator:
    def __init__(self):
        # 简单的启发式估计器（Phase 3.2.1）
        pass

    def estimate_eagle_payoff(self, logprob, depth):
        # Eagle: logprob 越高，越可能被接受
        # 简化：p_accept ≈ sigmoid(logprob)
        p_accept = 1 / (1 + exp(-logprob))
        return p_accept * depth

    def estimate_sam_payoff(self, match_length, depth):
        # SAM: match_length 越长，越可能被接受
        # 简化：p_accept ≈ match_length / max_length
        p_accept = min(match_length / 20, 1.0)
        return p_accept * depth

# 融合时使用 payoff 排序
for node in eagle_nodes:
    node.payoff = estimator.estimate_eagle_payoff(node.logprob, node.depth)

for node in sam_nodes:
    node.payoff = estimator.estimate_sam_payoff(node.match_length, node.depth)

# 按 payoff 排序选择
all_nodes = eagle_nodes + sam_nodes
selected = top_k(all_nodes, key=lambda n: n.payoff, k=60)
```

**预期效果**：
- 选择更有价值的候选
- 减少浪费的验证
- 提升整体 TPS

**目标**：> 56 TPS (1.001x)，超过 sam_sequence_graft

---

## 时间规划

### 本周

- **今天**：Run profiler
- **分析结果**：决定优化方向
- **如果场景 A**：设计 Phase 3.2 方案

### 下周

- **实现 Phase 3.2**：Payoff estimator
- **MT-Bench 验证**：目标 > 56 TPS
- **如果成功**：Phase 3.3 (context-aware)

### 2 周后

- **完整实验**：MT-Bench + MedQAd
- **论文初稿**：实验结果 + 分析

---

## 成功标准

### Phase 3.1 完成标准

- ✅ Profile 完成
- ✅ 瓶颈明确（有或无）
- ✅ 下一步方向清晰

### Phase 3 最终目标

```
Target: > 56.2 TPS (超过 sam_sequence_graft)
Method: Payoff-aware fusion
Innovation: 统一的期望收益框架
```

---

准备好在远端运行 profiler 了！把结果告诉我，我会立即给出下一步方案。🔍
