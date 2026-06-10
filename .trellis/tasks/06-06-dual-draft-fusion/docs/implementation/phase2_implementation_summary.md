# Phase 2 实现总结

## ✅ 已完成

### 核心模块 (`samd/fusion/`)

**1. `types.py`** - 数据结构定义
- `CandidateNode`: 单个候选节点（token, source, score, depth, path）
- `FusedTree`: 融合后的树（nodes, buffers, metadata）
- `FusionConfig`: 融合配置（mode, budget, dedup/truncate策略）

**2. `naive_fusion.py`** - 核心融合算法
- ✅ `parse_eagle_tree()`: 通过 `TreeSpec.from_eagle3_buffers()` 解析 Eagle3 树
- ✅ `parse_sam_sequence()`: SAM 线性序列转树结构
- ✅ `merge_and_dedup()`: 三种去重策略
  - `max_score`: 保留最高分数
  - `sum_score`: 合并分数（用于分析双引擎一致性）
  - `keep_both`: 不去重（消融实验用）
- ✅ `sort_by_score()`: 分源归一化后排序
- ✅ `truncate_with_ancestors()`: 前缀闭包截断
- ✅ `build_tree_buffers()`: 通过 `TreeSpec.to_buffers()` 构建验证 buffers

**3. `utils.py`** - 辅助函数
- ✅ `count_by_source()`: 统计各来源节点数

### 集成修改

**1. `samd/utils.py`**
- ✅ `gen_candidates()` 新增 `fusion_mode="naive"` 分支
- ✅ 并行生成 Eagle3 tree + SAM sequence
- ✅ 调用 `fuse_eagle_sam_naive()` 融合
- ✅ 返回兼容的 `Candidates` 对象

**2. `samd/samd_config.py`**
- ✅ 新增 `fusion_mode` 配置（目前支持 "none", "naive"）
- ✅ 新增 `fusion_max_draft_tokens`, `fusion_dedup_strategy`, `fusion_truncate_strategy`
- ✅ Config 验证：拒绝未实现的 fusion mode

**3. `samd/draft.py`**
- ✅ `DraftModel.record_naive_fusion()`: 记录融合统计
- ✅ 集成到现有的 `fusion_summary()` 输出

**4. CLI 集成**
- ✅ `evaluation/inference_samd.py`: 添加 `--fusion_mode` 等参数
- ✅ `samd/inference/cli.py`: 同步 CLI 参数

### 测试

**1. `tests/test_naive_fusion.py`**
- ✅ `test_parse_sam_sequence()`: SAM 解析测试
- ✅ `test_merge_and_dedup_max_score()`: 去重测试（max策略）
- ✅ `test_merge_and_dedup_sum_score()`: 去重测试（sum策略）
- ✅ `test_truncate_with_ancestors()`: 前缀闭包测试
- ✅ `test_build_tree_buffers()`: Buffer 构建测试
- ✅ `test_gen_candidates_naive_fusion()`: 集成烟雾测试
- ✅ `test_fusion_config_validation()`: 配置验证测试

**状态**: 本地无 `pytest`/`torch`，需在远程环境验证

**2. `evaluation/eval_naive_fusion.py`**
- ✅ HumanEval 评估脚本框架
- ✅ 三个 baseline：Eagle3-only, SAM-only, Naive fusion
- ✅ 自动计算 speedup, MAT, tokens/sec
- ✅ 输出融合统计（eagle/sam contribution, dedup rate）
- ✅ `--dry_run` 模式验证命令生成

**状态**: 脚本结构完成，需模型环境执行实际评估

### Trellis 任务管理

- ✅ 修复 `.trellis/tasks/06-06-dual-draft-fusion/task.json`
- ✅ 任务状态: `in_progress`
- ✅ 添加实现 metadata

## 📊 实现细节

### 评分策略（无 Eagle3 logprob）

由于 `Eagle3.gen_draft()` 当前不返回 logprob，采用简化策略：

**Eagle 节点**:
```python
score = 1.0 / (depth + 1)  # 深度越浅分数越高
```

**SAM 节点**:
```python
score = match_length - position  # 匹配长度 - 位置
```

**归一化**: 分别对 Eagle/SAM 节点归一化到 [0, 1]，然后排序

> **Phase 3 TODO**: 提取真实 logprob 用于 payoff-aware fusion

### 树结构解析（复用 TreeSpec）

**Eagle3 解析**:
```python
tree_spec = TreeSpec.from_eagle3_buffers(
    tokens, tree_attn_mask, tree_position_ids
)
# TreeSpec 自动推导父子关系
```

**SAM 解析**:
```python
# 线性序列 [t0, t1, t2, ...] 转树
# path[i] = [t0, t1, ..., t_{i-1}]
```

**Buffer 导出**:
```python
fused_spec = TreeSpec(tokens=..., parents=...)
buffers = fused_spec.to_buffers(device, dtype)
# 自动生成 tree_mask, position_ids, retrieve_indices
```

### 前缀闭包截断

算法确保：**选中子节点 → 所有祖先都被选中**

```python
def truncate_with_ancestors(sorted_nodes, max_tokens):
    selected = set()
    for node in sorted_nodes:
        ancestors = get_all_ancestors(node)
        if len(selected) + len(ancestors) + 1 <= max_tokens:
            selected.update(ancestors)
            selected.add(node)
    return list(selected)
```

### 融合统计输出

集成到现有的 `draft.fusion_summary()`：

```
Fusion Stats:
  eagle_nodes: 42
  sam_nodes: 30
  merged_nodes: 58 (dedup: 14)
  final_nodes: 60 (truncate: 0)
  eagle_contribution: 72.4%
  sam_contribution: 27.6%
```

## 🔍 验证状态

### ✅ 本地验证通过
- [x] Python 编译检查
- [x] 评估脚本 dry-run
- [x] 配置验证逻辑
- [x] CLI 参数完整性

### ⏳ 远程验证待执行
- [ ] Unit tests (需要 pytest + torch)
- [ ] HumanEval 实际评估（需要模型权重）
- [ ] 性能 profiling
- [ ] 与 Eagle3-only/SAM-only baseline 对比

## 📝 已知限制

### 1. 评分简化
- 当前使用深度作为 Eagle 分数代理
- Phase 3 需要提取真实 logprob

### 2. `keep_both` 去重策略
- 保留重复节点但不明确子节点归属
- 仅用于消融实验，主流程使用 `max_score`

### 3. 性能优化
- Naive 实现优先保证正确性
- Phase 3/4 将优化 GPU 利用率和并行度

## 🚀 下一步行动

### 立即执行（需模型环境）
1. **运行 unit tests**:
   ```bash
   pytest tests/test_naive_fusion.py -v
   ```

2. **HumanEval baseline 评估**:
   ```bash
   python evaluation/eval_naive_fusion.py \
     --model_path /path/to/llama-7b \
     --eagle3_path /path/to/eagle3 \
     --sam_path /path/to/sam.pkl \
     --bench_name humaneval \
     --num_choices 1
   ```

3. **验证融合效果**:
   - 检查 speedup: naive fusion > max(Eagle3, SAM)?
   - 检查 MAT: 融合是否增加接受 token 数?
   - 检查贡献度: Eagle/SAM 比例是否合理?

### Phase 3 准备（本周）
1. **提取 Eagle3 真实 logprob**
   - 修改 `Eagle3.gen_draft()` 返回分数
   - 或在 `parse_eagle_tree()` 中重新计算

2. **设计 Payoff 校准器**
   - Isotonic regression 实现
   - 在线更新机制

3. **实现 Context Feature Extraction**
   - Token 熵计算
   - 域检测（code/chat/summarization）
   - SAM 匹配质量估计

### 实验设计（下周）
1. **消融实验**:
   - Dedup: max_score vs sum_score vs keep_both
   - Truncate: score vs depth_first
   - Budget: 40 vs 60 vs 80

2. **场景对比**:
   - Code: HumanEval, RepoBench
   - Math: GSM8K
   - Summarization: CNN/DM

3. **分析维度**:
   - 双引擎一致性（agreement rate）
   - 去重率 vs 性能提升
   - 不同预算下的收益曲线

## 📂 文件清单

### 新增文件
```
samd/fusion/
├── __init__.py
├── types.py              (163 行)
├── naive_fusion.py       (437 行)
└── utils.py              (15 行)

tests/
└── test_naive_fusion.py  (312 行)

evaluation/
└── eval_naive_fusion.py  (202 行)

.trellis/tasks/06-06-dual-draft-fusion/
├── task.json             (Trellis metadata)
├── prd.md                (研究计划)
├── docs/research/
│   └── literature_review.md
└── docs/implementation/
    └── phase2_naive_baseline.md
```

### 修改文件
```
samd/
├── utils.py              (+68 行, fusion 分支)
├── samd_config.py        (+21 行, fusion config)
├── draft.py              (+15 行, stats recording)
└── inference/cli.py      (+9 行, CLI args)

evaluation/
└── inference_samd.py     (+9 行, CLI args)
```

## 🎯 成功标准

### Naive Baseline 达标条件
- [x] 代码实现完整（融合逻辑 + 集成 + 测试）
- [x] 配置系统扩展（CLI + config）
- [x] 评估框架搭建（baseline 对比脚本）
- [ ] Unit tests 通过（待远程验证）
- [ ] HumanEval 评估完成（待远程执行）
- [ ] 初步结果：naive fusion >= max(Eagle3, SAM)

### Phase 2 完成标准
- [ ] 至少 1 个场景下 naive fusion 显著优于单一 drafter
- [ ] 消融实验验证去重/截断策略有效性
- [ ] 融合统计分析（contribution, dedup rate）
- [ ] 为 Phase 3 提供 baseline 数据

## 💡 技术亮点

1. **复用 TreeSpec**: 避免重新实现复杂的 mask 构建逻辑
2. **前缀闭包保证**: 严格的祖先约束，确保树有效性
3. **灵活去重策略**: 支持 max/sum/keep_both，便于消融实验
4. **无侵入集成**: 不修改 `Candidates` 结构，复用现有统计通道
5. **渐进式验证**: 本地编译检查 + 远程运行时验证分离

## 📖 使用示例

### 基本用法
```bash
# Naive fusion
python evaluation/inference_samd.py \
  --model_path llama-7b \
  --fusion_mode naive \
  --fusion_dedup_strategy max_score \
  --fusion_max_draft_tokens 60
```

### 消融实验
```bash
# 测试不同去重策略
for dedup in max_score sum_score keep_both; do
  python evaluation/inference_samd.py \
    --fusion_mode naive \
    --fusion_dedup_strategy $dedup \
    --answer_file results_${dedup}.jsonl
done
```

### Baseline 对比
```bash
# 一键运行所有 baseline
python evaluation/eval_naive_fusion.py \
  --model_path llama-7b \
  --eagle3_path eagle3 \
  --sam_path sam.pkl \
  --bench_name humaneval
```

---

**总结**: Phase 2 Naive Baseline 实现完整，架构清晰，待远程环境验证。为 Phase 3 Payoff-Aware Fusion 打下坚实基础。
