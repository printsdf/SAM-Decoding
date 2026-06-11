# Phase 2.5 实施规格：添加统计 + 快速修复

## Part 1: 添加详细统计输出

### 修改 `samd/fusion/naive_fusion.py`

在 `fuse_eagle_sam_naive()` 函数中添加统计收集：

```python
def fuse_eagle_sam_naive(
    eagle_tree: Dict,
    sam_candidates: List[int],
    start_token: int,
    config: FusionConfig,
) -> FusedTree:
    # ... 现有代码 ...

    # 在返回前添加详细统计
    fusion_stats = {
        # 基础计数
        "eagle_nodes": len(eagle_nodes),
        "sam_nodes": len(sam_nodes),
        "merged_nodes": len(merged_nodes),
        "final_nodes": len(selected_nodes),

        # 分数分析
        "eagle_avg_score": _safe_mean([n.score for n in eagle_nodes]),
        "sam_avg_score": _safe_mean([n.score for n in sam_nodes]),
        "eagle_avg_depth": _safe_mean([n.depth for n in eagle_nodes]),
        "sam_avg_depth": _safe_mean([n.depth for n in sam_nodes]),

        # SAM 质量分析
        "sam_avg_match_length": _safe_mean([getattr(n, 'match_length', 0) for n in sam_nodes]),
        "sam_max_match_length": max([getattr(n, 'match_length', 0) for n in sam_nodes], default=0),

        # 去重分析
        "dedup_count": len(eagle_nodes) + len(sam_nodes) - len(merged_nodes),
        "both_proposed_count": _count_duplicates(eagle_nodes, sam_nodes),

        # 选择分析
        "truncated_count": len(merged_nodes) - len(selected_nodes),
        "selected_eagle": sum(1 for n in selected_nodes if n.source == "eagle"),
        "selected_sam": sum(1 for n in selected_nodes if n.source == "sam"),
    }

    fused_tree.metadata.update(fusion_stats)
    return fused_tree


def _safe_mean(values):
    """安全的平均值计算"""
    return sum(values) / len(values) if values else 0.0


def _count_duplicates(eagle_nodes, sam_nodes):
    """计算双引擎都提出的节点数"""
    eagle_keys = {(tuple(n.path), n.token) for n in eagle_nodes}
    sam_keys = {(tuple(n.path), n.token) for n in sam_nodes}
    return len(eagle_keys & sam_keys)
```

### 修改 `samd/draft.py`

在 `record_naive_fusion()` 中添加接受率统计：

```python
def record_naive_fusion(self, fusion_meta):
    """记录 naive fusion 统计"""
    # 现有统计
    self.fusion_stats["steps"] += 1
    self.fusion_stats["eagle_nodes_sum"] += fusion_meta.get("eagle_nodes", 0)
    self.fusion_stats["sam_nodes_sum"] += fusion_meta.get("sam_nodes", 0)

    # 新增：保存最近一次的完整统计
    self.fusion_stats["last_step"] = fusion_meta
```

在 `update()` 方法中添加验证后的接受统计：

```python
def update(self, tokens, last_hidden_states, verified_nodes=None, **kwargs):
    # ... 现有代码 ...

    # 如果有验证节点信息，统计接受率
    if verified_nodes is not None and self.config.fusion_mode == "naive":
        eagle_accepted = sum(1 for n in verified_nodes if n.get("source") == "eagle" and n.get("accepted"))
        sam_accepted = sum(1 for n in verified_nodes if n.get("source") == "sam" and n.get("accepted"))
        eagle_total = sum(1 for n in verified_nodes if n.get("source") == "eagle")
        sam_total = sum(1 for n in verified_nodes if n.get("source") == "sam")

        if "accept_rates" not in self.fusion_stats:
            self.fusion_stats["accept_rates"] = []

        self.fusion_stats["accept_rates"].append({
            "eagle_rate": eagle_accepted / eagle_total if eagle_total > 0 else 0,
            "sam_rate": sam_accepted / sam_total if sam_total > 0 else 0,
            "eagle_accepted": eagle_accepted,
            "sam_accepted": sam_accepted,
            "eagle_total": eagle_total,
            "sam_total": sam_total,
        })
```

### 修改 `samd/samd_model.py`

在 `generate()` 结束时输出聚合统计：

```python
def generate(self, ...):
    # ... 现有生成循环 ...

    # 输出聚合统计
    if self.gen_config.collect_diagnosis_trace and self.samd_config.fusion_mode == "naive":
        self._print_fusion_analysis()


def _print_fusion_analysis(self):
    """输出融合分析报告"""
    stats = self.draft.fusion_stats
    if not stats or stats.get("steps", 0) == 0:
        return

    steps = stats["steps"]
    print("\n" + "="*60)
    print("Naive Fusion Analysis")
    print("="*60)

    # 候选节点统计
    avg_eagle = stats["eagle_nodes_sum"] / steps
    avg_sam = stats["sam_nodes_sum"] / steps
    print(f"Average nodes per step:")
    print(f"  Eagle: {avg_eagle:.1f}")
    print(f"  SAM:   {avg_sam:.1f}")
    print(f"  Ratio: {avg_sam / (avg_eagle + avg_sam):.1%} SAM")

    # 接受率统计
    if "accept_rates" in stats and stats["accept_rates"]:
        rates = stats["accept_rates"]
        avg_eagle_rate = sum(r["eagle_rate"] for r in rates) / len(rates)
        avg_sam_rate = sum(r["sam_rate"] for r in rates) / len(rates)
        total_eagle_accepted = sum(r["eagle_accepted"] for r in rates)
        total_sam_accepted = sum(r["sam_accepted"] for r in rates)

        print(f"\nAcceptance rates:")
        print(f"  Eagle: {avg_eagle_rate:.1%}")
        print(f"  SAM:   {avg_sam_rate:.1%}")
        print(f"  Delta: {avg_eagle_rate - avg_sam_rate:+.1%}")

        print(f"\nTotal accepted tokens:")
        print(f"  Eagle: {total_eagle_accepted}")
        print(f"  SAM:   {total_sam_accepted}")
        print(f"  SAM contribution: {total_sam_accepted / (total_eagle_accepted + total_sam_accepted):.1%}")

    # 最后一步详细信息
    if "last_step" in stats:
        last = stats["last_step"]
        print(f"\nLast step details:")
        print(f"  SAM avg match length: {last.get('sam_avg_match_length', 0):.1f}")
        print(f"  SAM max match length: {last.get('sam_max_match_length', 0)}")
        print(f"  Both proposed: {last.get('both_proposed_count', 0)}")
        print(f"  Dedup removed: {last.get('dedup_count', 0)}")

    print("="*60)
```

---

## Part 2: 快速修复（基于统计结果）

### 方案 A：添加质量门控

修改 `samd/utils.py::gen_candidates()`：

```python
def gen_candidates(...):
    # ... 现有代码 ...

    elif samd_config.fusion_mode == "naive":
        start_token = sample_p.squeeze(0).argmax(-1).item()

        # 新增：检查 SAM 匹配质量
        index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
        index_static, match_static = draft.sam_static.lookup(start_token)
        best_match = max(match_dyn, match_static - draft.len_bias)

        # 质量门控：SAM 质量差时只用 Eagle
        if best_match < samd_config.samd_len_threshold:
            # 退化到 Eagle-only
            eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
            draft.record_naive_fusion({
                "eagle_nodes": len(eagle_pred_ids),
                "sam_nodes": 0,
                "sam_skipped": True,
                "sam_match_quality": best_match,
            })
            return Candidates(
                type=CandidateType.tree,
                tokens=torch.tensor([eagle_pred_ids], device=device),
                candidate_tokens=extract_candidate_tokens_from_eagle(eagle_pred_ids, eagle_buffers),
                buffers_kwargs=eagle_buffers,
            )

        # SAM 质量好，进行融合
        # ... 现有融合逻辑 ...
```

### 方案 B：保护 Eagle 优先级

修改 `samd/fusion/naive_fusion.py::merge_and_dedup()`：

```python
def merge_and_dedup(
    eagle_nodes: List[CandidateNode],
    sam_nodes: List[CandidateNode],
    strategy: str,
    eagle_boost: float = 1.2,  # 新增参数
) -> List[CandidateNode]:
    """
    合并去重，保护 Eagle 优先级

    Args:
        eagle_boost: Eagle 分数加成因子（>1.0 提升 Eagle 优先级）
    """
    # 应用 Eagle boost
    for node in eagle_nodes:
        node.boosted_score = node.normalized_score * eagle_boost

    for node in sam_nodes:
        node.boosted_score = node.normalized_score

    # 按 (path, token) 构建字典
    merged_dict = {}

    for node in eagle_nodes:
        key = (tuple(node.path), node.token)
        merged_dict[key] = node

    for node in sam_nodes:
        key = (tuple(node.path), node.token)
        if key in merged_dict:
            # 重复节点：比较 boosted_score
            if strategy == "max_score":
                if node.boosted_score > merged_dict[key].boosted_score:
                    merged_dict[key] = node
            elif strategy == "sum_score":
                # 合并分数，保留 Eagle 来源
                merged_dict[key].score += node.score
                merged_dict[key].boosted_score += node.boosted_score
        else:
            merged_dict[key] = node

    return list(merged_dict.values())
```

修改 `samd/fusion/naive_fusion.py::truncate_with_ancestors()`：

```python
def truncate_with_ancestors(
    sorted_nodes: List[CandidateNode],
    max_tokens: int,
    min_eagle_ratio: float = 0.7,  # 新增参数
) -> List[CandidateNode]:
    """
    前缀闭包截断，确保最少 Eagle 比例
    """
    # 第一轮：贪心选择（使用 boosted_score）
    selected_keys = set()
    selected_nodes = []

    for node in sorted_nodes:
        # 收集祖先
        ancestors = _collect_ancestors(node, all_nodes_dict)
        needed_keys = {(tuple(a.path), a.token) for a in ancestors} | {(tuple(node.path), node.token)}

        # 检查是否超预算
        if len(selected_keys | needed_keys) <= max_tokens:
            selected_keys.update(needed_keys)
            selected_nodes.append(node)

    # 第二轮：确保 Eagle 比例
    eagle_count = sum(1 for n in selected_nodes if n.source == "eagle")
    eagle_ratio = eagle_count / len(selected_nodes) if selected_nodes else 0

    if eagle_ratio < min_eagle_ratio:
        # Eagle 太少，强制替换一些 SAM 节点
        target_eagle = int(len(selected_nodes) * min_eagle_ratio)
        missing_eagle = target_eagle - eagle_count

        # 找到被排除的 Eagle 节点（按 boosted_score 排序）
        excluded_eagle = [n for n in sorted_nodes if n.source == "eagle" and n not in selected_nodes]

        # 找到可以移除的 SAM 节点（按 boosted_score 逆序）
        selected_sam = [n for n in selected_nodes if n.source == "sam"]
        selected_sam.sort(key=lambda n: n.boosted_score)

        # 替换
        for i in range(min(missing_eagle, len(excluded_eagle), len(selected_sam))):
            selected_nodes.remove(selected_sam[i])
            selected_nodes.append(excluded_eagle[i])

    return selected_nodes
```

---

## Part 3: 配置扩展

修改 `samd/samd_config.py`：

```python
@dataclass
class SamdConfig:
    # ... 现有字段 ...

    # Fusion 相关配置
    fusion_mode: Literal["none", "naive", "payoff_aware", "tree_aware"] = field(default="none")
    fusion_max_draft_tokens: int = field(default=60)
    fusion_dedup_strategy: Literal["max_score", "sum_score", "keep_both"] = field(default="max_score")

    # 新增配置
    fusion_eagle_boost: float = field(default=1.2)  # Eagle 分数加成
    fusion_min_eagle_ratio: float = field(default=0.7)  # 最小 Eagle 比例
    fusion_use_quality_gate: bool = field(default=True)  # 是否使用质量门控
```

---

## 实施步骤

### Step 1: 添加统计（今天）
1. 实现 Part 1 的所有修改
2. 编译测试
3. 在 HumanEval 上跑 10 个样本验证统计输出

### Step 2: 分析数据（今天）
1. 查看聚合统计报告
2. 分析：
   - Eagle vs SAM 接受率差异
   - SAM 平均匹配长度
   - 双引擎一致性（both_proposed_count）
3. 确定修复优先级

### Step 3: 实施修复（明天）
1. 基于数据决定：
   - 是否需要质量门控（方案 A）
   - Eagle boost 和 min_ratio 的合适值（方案 B）
2. 实现修复
3. 重新评估完整 HumanEval

### Step 4: 对比验证（明天）
```bash
# 修复前
naive_fusion (原始): 5.915 MAT, 49.481 TPS (0.816x)

# 目标：修复后
naive_fusion (v2): > 6.5 MAT, > 54 TPS (> 0.9x)

# 上限参考
sam_sequence_graft: 7.304 MAT, 63.098 TPS (1.041x)
```

---

## 预期输出示例

### 统计报告
```
============================================================
Naive Fusion Analysis
============================================================
Average nodes per step:
  Eagle: 42.3
  SAM:   18.7
  Ratio: 30.7% SAM

Acceptance rates:
  Eagle: 62.4%
  SAM:   23.1%
  Delta: +39.3%

Total accepted tokens:
  Eagle: 526
  SAM:   87
  SAM contribution: 14.2%

Last step details:
  SAM avg match length: 8.5
  SAM max match length: 15
  Both proposed: 3
  Dedup removed: 8
============================================================
```

### 关键指标
- **如果 SAM accept_rate < 30%**：证明 SAM 质量差，需要质量门控
- **如果 both_proposed < 5**：证明双引擎一致性差，分数融合有问题
- **如果 sam_avg_match_length < 5**：证明动态 SAM 匹配质量差

---

## 成功标准

**Part 1 完成**：
- [x] 统计代码实现
- [x] 编译通过
- [x] 输出格式正确

**Part 2 完成**：
- [x] 修复实现
- [x] HumanEval 完整评估
- [x] naive_fusion 提升到 > 0.9x Eagle3

**最终目标**：
- naive_fusion (v2) 作为可靠的 baseline
- 为 Phase 3 (payoff-aware) 打好基础
