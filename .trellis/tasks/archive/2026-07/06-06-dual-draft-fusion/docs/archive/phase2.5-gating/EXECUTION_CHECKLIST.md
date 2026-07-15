# Phase 2.5 远端执行清单

## 已完成（本地）

✅ **统计代码实现**
- `samd/fusion/naive_fusion.py`: 添加详细 fusion_stats
- `samd/draft.py`: 记录接受率统计
- `samd/samd_model.py`: 打印分析报告
- `tests/test_naive_fusion.py`: 新增回归测试
- ✅ 编译检查通过

## 待执行（远端）

### Step 1: 验证统计输出（10 样本测试）

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 安装依赖（如果需要）
# pip install pytest -q

# 运行单元测试
pytest tests/test_naive_fusion.py -v

# 运行 10 样本 HumanEval 验证统计
python evaluation/inference_samd.py \
  --model_type llama \
  --model_path /teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
  --model_id llama31-8b-instruct \
  --tree_model_path /teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
  --bench_name humaneval \
  --question_begin 0 \
  --question_end 10 \
  --num_choices 1 \
  --max_new_tokens 512 \
  --dtype bfloat16 \
  --fusion_mode naive \
  --fusion_dedup_strategy max_score \
  --fusion_max_draft_tokens 60 \
  --answer_file humaneval_naive_stats_test_10samples.jsonl

# 查看输出的统计报告（会在 stdout 末尾）
# 关注：
# - Eagle vs SAM 接受率
# - SAM 平均匹配长度
# - 双引擎一致性（Both proposed）
```

### Step 2: 分析统计结果

**关键指标判断**：

```python
# 如果看到：
Eagle acceptance rate: ~60-70%
SAM acceptance rate: ~20-30%
Delta: +30-40%
SAM avg match length: < 5

# 结论：SAM 质量差，需要质量门控（方案 A）
```

```python
# 如果看到：
Both proposed count: < 5
Dedup removed: > 10
Selected SAM / Final nodes: > 40%

# 结论：分数融合不准确，需要 Eagle 优先级保护（方案 B）
```

### Step 3: 实施修复（基于统计结果）

#### 方案 A：质量门控（如果 SAM accept rate < 30%）

修改 `samd/utils.py::gen_candidates()`，在 `fusion_mode == "naive"` 分支添加：

```python
# 在融合前检查 SAM 质量
index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
index_static, match_static = draft.sam_static.lookup(start_token)
best_match = max(match_dyn, match_static - draft.len_bias)

# 质量门控
if best_match < samd_config.samd_len_threshold:
    # SAM 质量差 → 只用 Eagle
    eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
    # ... 返回 Eagle-only candidates
    return eagle_only_result

# SAM 质量好 → 继续融合
# ... 现有融合逻辑
```

#### 方案 B：Eagle 优先级（如果 SAM 挤掉了高质量 Eagle）

修改 `samd/fusion/naive_fusion.py`：

```python
# 在 merge_and_dedup() 中添加 eagle_boost 参数
for node in eagle_nodes:
    node.boosted_score = node.normalized_score * 1.2  # Eagle boost

# 在 truncate_with_ancestors() 中添加 min_eagle_ratio
# 确保最终至少 70% Eagle 节点
```

或者直接修改配置：

```bash
# 测试不同 boost 因子
python evaluation/inference_samd.py \
  --fusion_mode naive \
  --fusion_eagle_boost 1.2 \
  --fusion_min_eagle_ratio 0.7 \
  ...
```

### Step 4: 完整评估（修复后）

```bash
# 修复后重新跑完整 HumanEval
python evaluation/eval_naive_fusion.py \
  --model_path /teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
  --tree_model_path /teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
  --bench_name humaneval \
  --num_choices 1 \
  --answer_dir naive_fusion_v2_20260607

# 对比结果
# 期望：
# naive_fusion_v2: > 6.5 MAT, > 54 TPS (> 0.9x Eagle3)
#
# 上限参考：
# sam_sequence_graft: 7.304 MAT, 63.098 TPS (1.041x Eagle3)
```

---

## 预期统计输出示例

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
  Delta: +39.3%  ← 如果 > +30%，说明 SAM 质量差

Total accepted tokens:
  Eagle: 526
  SAM:   87
  SAM contribution: 14.2%  ← 如果 < 20%，说明 SAM 没有有效贡献

Last step details:
  SAM avg match length: 8.5  ← 如果 < 5，说明动态 SAM 匹配质量差
  SAM max match length: 15
  Both proposed: 3  ← 如果 < 5，说明双引擎一致性差
  Dedup removed: 8
============================================================
```

---

## 决策树

```
统计分析
    │
    ├─ SAM accept rate < 30%？
    │   └─ YES → 实施方案 A（质量门控）
    │
    ├─ SAM contribution < 20%？
    │   └─ YES → 实施方案 A（质量门控）
    │
    ├─ Both proposed < 5？
    │   └─ YES → 实施方案 B（Eagle 优先级）
    │
    └─ Selected SAM > 40%？
        └─ YES → 实施方案 B（Eagle 优先级）
```

---

## 成功标准

### Phase 2.5 完成：
- [x] 统计代码实现（本地完成）
- [ ] 统计输出验证（远端 10 样本）
- [ ] 修复实施（基于统计结果）
- [ ] 完整评估（HumanEval 164 样本）
- [ ] naive_fusion 提升到 > 0.9x Eagle3

### 最终目标：
- naive_fusion (v2) 作为可靠的 baseline
- 为 Phase 3 (payoff-aware) 打好基础
- 论文故事：Naive 失败 → Payoff-aware 成功

---

## 备注

- 代码已同步到远端（需要 pull 最新代码）
- 本地无法运行需要 torch/pytest 的测试
- 所有模型相关验证需要在远端执行
- 统计输出会在生成完成后打印到 stdout

---

## 快速命令（复制粘贴）

```bash
# 拉取最新代码
cd /teamspace/studios/this_studio/SAM-Decoding
git pull  # 如果代码已推送

# 或直接同步修改的文件
# rsync 或手动复制 samd/fusion/naive_fusion.py, samd/draft.py, samd/samd_model.py

# 运行 10 样本测试
python evaluation/inference_samd.py \
  --model_type llama \
  --model_path /teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
  --tree_model_path /teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
  --bench_name humaneval \
  --question_begin 0 --question_end 10 \
  --num_choices 1 --max_new_tokens 512 --dtype bfloat16 \
  --fusion_mode naive \
  --answer_file test_stats_10.jsonl 2>&1 | tee test_stats_10.log

# 查看统计
tail -50 test_stats_10.log
```
