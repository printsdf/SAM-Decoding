# Phase 2: Naive Baseline 实现规格

## 目标

实现最简单的 Eagle3 + SAM 融合策略，作为后续优化的 baseline。

## 架构设计

### 1. 模块结构

```
samd/fusion/
├── __init__.py
├── types.py          # 数据结构定义
├── naive_fusion.py   # Naive fusion 实现
└── utils.py          # 辅助函数
```

### 2. 核心数据结构

```python
# samd/fusion/types.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
import torch

@dataclass
class CandidateNode:
    """单个候选节点"""
    token: int                    # token ID
    source: Literal["eagle", "sam"]  # 来源
    score: float                  # 原始分数（Eagle: logprob, SAM: match_length）
    depth: int                    # 深度
    path: List[int]              # 从 root 到当前节点的路径
    parent_idx: Optional[int]    # 父节点索引（在融合树中）

@dataclass
class FusedTree:
    """融合后的候选树"""
    nodes: List[CandidateNode]         # 所有节点
    tokens: torch.Tensor               # [N] token IDs
    tree_mask: torch.Tensor            # [N, N] attention mask
    position_ids: torch.Tensor         # [N] position IDs
    retrieve_indices: torch.Tensor     # [M] retrieve indices for candidate tokens
    metadata: Dict                     # 统计信息

@dataclass
class FusionConfig:
    """融合配置"""
    mode: Literal["naive", "payoff_aware", "tree_aware"] = "naive"
    max_draft_tokens: int = 60         # 总预算
    dedup_strategy: Literal["max_score", "sum_score", "keep_both"] = "max_score"
    truncate_strategy: Literal["score", "depth_first"] = "score"
```

### 3. Naive Fusion 算法

```python
# samd/fusion/naive_fusion.py

from typing import Tuple, Dict
import torch
from .types import CandidateNode, FusedTree, FusionConfig

def fuse_eagle_sam_naive(
    eagle_tree: Dict,              # Eagle3 生成的树
    sam_candidates: List[int],     # SAM 生成的序列
    start_token: int,
    config: FusionConfig,
) -> FusedTree:
    """
    Naive fusion: 简单合并 + 去重 + 截断

    算法步骤:
    1. 解析 Eagle tree 为 node list
    2. 解析 SAM sequence 为 node list
    3. 合并：相同 (path, token) 保留最高 score
    4. 排序：按 score 降序
    5. 截断：保留前 max_draft_tokens 个
    6. 构建 tree buffers

    Args:
        eagle_tree: {
            "tokens": torch.Tensor,  # [N_eagle]
            "tree_mask": torch.Tensor,
            "position_ids": torch.Tensor,
            "retrieve_indices": torch.Tensor,
        }
        sam_candidates: [start_token, t1, t2, ...]  # 线性序列
        start_token: root token
        config: 融合配置

    Returns:
        FusedTree: 融合后的树结构
    """
    # Step 1: 解析 Eagle tree
    eagle_nodes = parse_eagle_tree(eagle_tree, start_token)

    # Step 2: 解析 SAM sequence
    sam_nodes = parse_sam_sequence(sam_candidates, start_token)

    # Step 3: 合并去重
    merged_nodes = merge_and_dedup(
        eagle_nodes,
        sam_nodes,
        strategy=config.dedup_strategy
    )

    # Step 4: 排序
    sorted_nodes = sort_by_score(merged_nodes)

    # Step 5: 截断
    if len(sorted_nodes) > config.max_draft_tokens:
        selected_nodes = truncate_with_ancestors(
            sorted_nodes,
            config.max_draft_tokens
        )
    else:
        selected_nodes = sorted_nodes

    # Step 6: 构建 tree buffers
    fused_tree = build_tree_buffers(selected_nodes, start_token)

    return fused_tree


def parse_eagle_tree(eagle_tree: Dict, start_token: int) -> List[CandidateNode]:
    """
    将 Eagle3 的树结构解析为 node list

    Eagle3 输出格式:
    - tokens: [N] flattened tree tokens
    - tree_mask: [N, N] attention mask (lower triangular + tree structure)
    - position_ids: [N] relative positions
    - retrieve_indices: [M] indices to extract candidate logits

    需要反向推导:
    - 从 tree_mask 推导父子关系
    - 从 retrieve_indices 推导哪些是 candidate nodes
    - 计算每个 node 的路径和深度

    Returns:
        List[CandidateNode]: 包含所有 Eagle nodes
    """
    pass  # 实现细节


def parse_sam_sequence(sam_candidates: List[int], start_token: int) -> List[CandidateNode]:
    """
    将 SAM 的线性序列解析为 node list

    SAM 输出格式:
    - [start_token, t1, t2, ..., tn]  # 简单的线性序列

    转换为树结构:
    - 每个 token 的 path = [start_token, t1, ..., t_{i-1}]
    - depth = index
    - score = len(sam_candidates) - index  (简单的匹配长度估计)

    Returns:
        List[CandidateNode]: 包含所有 SAM nodes
    """
    nodes = []
    path = [start_token]
    for i, token in enumerate(sam_candidates[1:], start=1):  # 跳过 start_token
        nodes.append(CandidateNode(
            token=token,
            source="sam",
            score=len(sam_candidates) - i,  # 越靠前分数越高
            depth=i,
            path=path.copy(),
            parent_idx=None,  # 稍后填充
        ))
        path.append(token)
    return nodes


def merge_and_dedup(
    eagle_nodes: List[CandidateNode],
    sam_nodes: List[CandidateNode],
    strategy: str,
) -> List[CandidateNode]:
    """
    合并并去重

    去重规则:
    - 相同 (path, token) 的节点视为重复
    - max_score: 保留分数最高的
    - sum_score: 合并分数并标记来源
    - keep_both: 保留两者（不去重，用于消融实验）

    Returns:
        List[CandidateNode]: 去重后的节点列表
    """
    pass  # 实现细节


def sort_by_score(nodes: List[CandidateNode]) -> List[CandidateNode]:
    """
    按分数降序排序

    注意: Eagle 的 score 是 logprob (负数)，SAM 的 score 是 match_length (正数)
    需要归一化到统一尺度

    简单做法: 分别归一化到 [0, 1]
    """
    pass  # 实现细节


def truncate_with_ancestors(
    sorted_nodes: List[CandidateNode],
    max_tokens: int,
) -> List[CandidateNode]:
    """
    截断到 max_tokens，同时保证祖先节点存在（前缀闭包）

    算法:
    1. 贪心选择高分节点
    2. 如果选中一个节点，确保其所有祖先都被选中
    3. 如果加上祖先会超预算，跳过该节点

    Returns:
        List[CandidateNode]: 截断后的节点列表（保证前缀闭包）
    """
    pass  # 实现细节


def build_tree_buffers(
    nodes: List[CandidateNode],
    start_token: int,
) -> FusedTree:
    """
    从 node list 构建树的 attention mask, position_ids, retrieve_indices

    输出格式需要与 Eagle3 兼容，以便直接输入到 target model 验证

    Returns:
        FusedTree: 包含所有必要的 buffers
    """
    pass  # 实现细节
```

### 4. 集成到 SAMD

```python
# 修改 samd/utils.py

from .fusion.naive_fusion import fuse_eagle_sam_naive
from .fusion.types import FusionConfig

def gen_candidates(
    sample_p: torch.Tensor,
    base_tree_retrieve_indices: torch.Tensor,
    draft: DraftModel,
    samd_config: SamdConfig,
    gen_config: SamdGenerationConfig,
    device: str,
):
    """
    生成候选 tokens

    新增参数:
    - samd_config.fusion_mode: "none" | "naive" | "payoff_aware" | "tree_aware"
    - samd_config.fusion_config: FusionConfig
    """
    start_token = sample_p.squeeze(0).argmax(-1).item()

    # 检查是否启用 fusion
    if samd_config.fusion_mode == "none":
        # 原有逻辑：根据 SAM 匹配质量决定用 Eagle 还是 SAM
        # ... (保持不变)
        pass

    elif samd_config.fusion_mode == "naive":
        # Naive fusion: 并行生成 Eagle + SAM，然后合并

        # Step 1: 生成 Eagle tree
        eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
        eagle_tree = {
            "tokens": torch.tensor(eagle_pred_ids, device=device),
            **eagle_buffers,
        }

        # Step 2: 生成 SAM sequence
        sam_index, sam_length = draft.sam.lookup(start_token)
        sam_candidates = draft.sam.gen_draft_raw(
            sam_index,
            start_token,
            max_len=samd_config.n_predicts
        )

        # Step 3: 融合
        fusion_config = samd_config.fusion_config or FusionConfig()
        fused_tree = fuse_eagle_sam_naive(
            eagle_tree=eagle_tree,
            sam_candidates=sam_candidates,
            start_token=start_token,
            config=fusion_config,
        )

        # Step 4: 构建返回值
        return Candidates(
            type=CandidateType.tree,
            tokens=fused_tree.tokens.unsqueeze(0),
            candidate_tokens=extract_candidate_tokens(fused_tree),
            buffers_kwargs={
                "tree_attn_mask": fused_tree.tree_mask,
                "tree_position_ids": fused_tree.position_ids,
                "tree_retrieve_indices": fused_tree.retrieve_indices,
            },
            metadata=fused_tree.metadata,
        )

    else:
        raise ValueError(f"Unknown fusion_mode: {samd_config.fusion_mode}")
```

### 5. 配置文件扩展

```python
# samd/samd_config.py

@dataclass
class SamdConfig:
    # ... 现有字段 ...

    # Fusion 相关配置
    fusion_mode: Literal["none", "naive", "payoff_aware", "tree_aware"] = "none"
    fusion_max_draft_tokens: int = 60
    fusion_dedup_strategy: Literal["max_score", "sum_score", "keep_both"] = "max_score"
    fusion_truncate_strategy: Literal["score", "depth_first"] = "score"

    @property
    def fusion_config(self) -> Optional[FusionConfig]:
        if self.fusion_mode == "none":
            return None
        return FusionConfig(
            mode=self.fusion_mode,
            max_draft_tokens=self.fusion_max_draft_tokens,
            dedup_strategy=self.fusion_dedup_strategy,
            truncate_strategy=self.fusion_truncate_strategy,
        )
```

## 实现优先级

### P0 (必须实现)
- [ ] `samd/fusion/types.py`: 数据结构定义
- [ ] `samd/fusion/naive_fusion.py`: 核心融合逻辑
  - [ ] `parse_sam_sequence()` - 简单
  - [ ] `parse_eagle_tree()` - 复杂，需要反推树结构
  - [ ] `merge_and_dedup()` - 中等
  - [ ] `sort_by_score()` - 简单
  - [ ] `truncate_with_ancestors()` - 中等
  - [ ] `build_tree_buffers()` - 复杂，需要构建 attention mask
- [ ] 集成到 `samd/utils.py::gen_candidates()`
- [ ] 扩展 `samd/samd_config.py`

### P1 (可选优化)
- [ ] `samd/fusion/utils.py`: 辅助函数
  - [ ] `visualize_tree()`: 可视化融合树
  - [ ] `compute_statistics()`: 计算融合统计信息
- [ ] 诊断工具：输出每步的融合 metadata

## 测试计划

### Unit Tests
```python
# tests/test_naive_fusion.py

def test_parse_sam_sequence():
    """测试 SAM 序列解析"""
    pass

def test_merge_and_dedup():
    """测试去重逻辑"""
    pass

def test_truncate_with_ancestors():
    """测试前缀闭包截断"""
    pass

def test_build_tree_buffers():
    """测试 buffer 构建"""
    pass
```

### Integration Tests
```python
# tests/test_fusion_integration.py

def test_gen_candidates_naive_fusion():
    """测试集成到 gen_candidates"""
    pass

def test_full_decode_with_fusion():
    """测试完整的 decode 流程"""
    pass
```

### Evaluation Script
```python
# evaluation/eval_naive_fusion.py

# 对比实验：
# 1. Eagle3-only
# 2. SAM-only
# 3. Naive fusion (max_score)
# 4. Naive fusion (sum_score)
# 5. Naive fusion (keep_both)

# 数据集: HumanEval (small, fast iteration)
# 指标: speedup, MAT, verified_per_accepted, eagle_contrib, sam_contrib
```

## Codex 实现指令

请按以下顺序实现：

1. **创建模块结构**
   - 创建 `samd/fusion/` 目录
   - 创建 `__init__.py`, `types.py`, `naive_fusion.py`, `utils.py`

2. **实现 types.py**
   - 定义 `CandidateNode`, `FusedTree`, `FusionConfig`

3. **实现 naive_fusion.py 的简单函数**
   - `parse_sam_sequence()` - 线性转树
   - `sort_by_score()` - 排序

4. **实现复杂函数**
   - `parse_eagle_tree()` - 需要仔细研究 Eagle3 的输出格式
   - `merge_and_dedup()` - 去重逻辑
   - `truncate_with_ancestors()` - 前缀闭包

5. **实现 build_tree_buffers()**
   - 这是最复杂的部分，需要构建兼容的 attention mask
   - 参考 Eagle3 的实现

6. **集成到 SAMD**
   - 修改 `samd/utils.py::gen_candidates()`
   - 扩展 `samd/samd_config.py`

7. **测试**
   - 编写 unit tests
   - 在 HumanEval 上运行简单测试

## 注意事项

1. **Eagle3 树结构解析是关键难点**
   - 需要仔细阅读 `samd/tree_model/eagle3/eagle3.py::gen_draft()` 的输出格式
   - 特别注意 `tree_mask` 如何编码父子关系
   - 参考 `retrieve_indices` 理解候选节点的位置

2. **分数归一化**
   - Eagle logprob: 通常在 [-20, 0] 范围
   - SAM match_length: 通常在 [0, 40] 范围
   - 需要归一化到统一尺度再比较

3. **前缀闭包约束**
   - 必须严格保证：选中子节点 → 所有祖先都被选中
   - 否则 attention mask 会出错

4. **性能考虑**
   - Naive 实现先不考虑性能，保证正确性
   - 后续 Phase 3/4 再优化

5. **诊断信息**
   - 在 metadata 中记录详细统计：
     - eagle_nodes, sam_nodes, merged_nodes, final_nodes
     - dedup_count, truncate_count
     - eagle_contribution, sam_contribution
   - 便于后续分析

## 预期输出

实现完成后，应该能够：

```python
# 运行示例
python evaluation/eval_naive_fusion.py \
  --model llama-7b \
  --dataset humaneval \
  --fusion_mode naive \
  --dedup_strategy max_score

# 输出示例
Results on HumanEval:
- Eagle3-only:     speedup=2.5x, MAT=5.2
- SAM-only:        speedup=1.8x, MAT=3.1
- Naive fusion:    speedup=2.7x, MAT=5.8
  - Eagle contrib: 62%
  - SAM contrib:   38%
  - Dedup rate:    15%
```
