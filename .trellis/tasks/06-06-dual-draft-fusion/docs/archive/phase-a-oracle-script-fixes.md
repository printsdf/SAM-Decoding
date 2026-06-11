# Phase A Oracle Scripts - 修复记录

## 2026-06-10 首次运行错误

### 错误 1: oracle_rejection_boundary.py
**问题**: `TypeError: selected_mat() got an unexpected keyword argument 'source'`

**原因**: `selected_mat(step, source="eagle")` 调用错误，该函数签名是 `selected_mat(step, selected_candidates)`

**修复**:
```python
# 错误
eagle_mat = selected_mat(step, source="eagle")

# 正确
eagle_candidates = [c for c in step.candidates if c.source == "eagle"]
eagle_accepted = [c for c in eagle_candidates if candidate_matches_acceptance(c, step)]
eagle_mat = selected_mat(step, eagle_accepted)
```

### 错误 2: oracle_high_precision_sam.py
**问题**: 所有阈值结果 gap = 0.00%，SAM count = 0.00

**原因**: SAM 候选没有 `match_length` 字段，`hasattr(c, 'match_length')` 总是返回 False

**修复**: 使用 `c.depth` 作为 match quality 的代理指标（depth 越深表示 SAM 匹配越长）

```python
# 错误
filtered_sam = [
    c for c in sam_candidates
    if hasattr(c, 'match_length') and c.match_length >= min_match_length
]

# 正确
filtered_sam = [
    c for c in sam_candidates
    if c.depth >= min_match_length
]
```

**说明**: SAM depth 反映了从 cache 匹配的长度，深度越大说明匹配质量越高

---

## 下一步

修复后重新运行：

```bash
# 1. Rejection-Boundary 分析（已修复 selected_mat 调用）
python evaluation/oracle_rejection_boundary.py \
  --trace-file evaluation/data/mt_bench/profile_fusion_overhead/naive_fusion_mt_bench_q0_80.fusion_profile.json \
  --output evaluation/data/mt_bench/profile_fusion_overhead/mt_bench_rejection_boundary.json

# 2. High-Precision SAM 分析（已修复 match_length 过滤）
python evaluation/oracle_high_precision_sam.py \
  --trace-file evaluation/data/mt_bench/profile_fusion_overhead/naive_fusion_mt_bench_q0_80.fusion_profile.json \
  --output evaluation/data/mt_bench/profile_fusion_overhead/mt_bench_high_precision_sam.json
```

期待看到真实的 oracle gap 数据！
