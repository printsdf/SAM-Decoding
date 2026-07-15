# Phase 2 Boundary Graft Implementation Complete

**Date**: 2026-06-11
**Status**: Ready for testing
**Task**: `.trellis/tasks/06-06-dual-draft-fusion`

---

## Implementation Summary

### Core Algorithm

Phase 2 实现了**预测边界 SAM 接力**（Predicted-Boundary SAM Grafting）：

1. **EAGLE drafting** with logprobs
2. **Predict rejection depth** from per-depth confidence (exp of logprob)
3. **Extract EAGLE prefix** up to predicted depth
4. **SAM continuation** from that prefix using transfer_state()
5. **Graft SAM tail** onto EAGLE tree at predicted node
6. **Single verification** pass

### Files Modified

#### New Files
1. **`samd/fusion/boundary_graft.py`** (~140 lines)
   - `predict_rejection_depth()`: Predict depth from confidence
   - `extract_prefix_tokens_and_node()`: Get prefix + graft point
   - `graft_sam_at_depth()`: Graft SAM onto specific node

2. **`scripts/eval_boundary_graft.sh`**: Single-run evaluation
3. **`scripts/eval_boundary_graft_sweep.sh`**: Threshold sweep

#### Modified Files
1. **`samd/samd_config.py`**: Added config fields
   - `boundary_graft_threshold` (default 1.5)
   - `boundary_graft_max_sam_nodes` (default 8)
   - `boundary_graft_min_depth` (default 3)
   - `boundary_graft_max_depth` (default 8)

2. **`samd/utils.py`**: Added `fusion_mode="boundary_graft"` branch (~180 lines)

3. **`evaluation/inference_samd.py`**: Added CLI arguments

---

## Critical Fixes (from Gemini Review)

### Bug 1: Confidence Metric Fixed ✅
- **Problem**: EAGLE logprobs are negative, threshold comparison was wrong
- **Fix**: Convert logprob to probability with `exp()`, threshold now in [0, 1]

### Bug 2: Path Consistency Fixed ✅
- **Problem**: Prefix extraction and graft point were decoupled
- **Fix**: `extract_prefix_tokens_and_node()` returns both prefix and node index

### Bug 3: Graft Depth Fixed ✅
- **Problem**: Off-by-one error in depth calculation
- **Fix**: Graft onto `parent_node_idx` directly, no depth-based search

### Bug 4: Leaf Constraint Removed ✅
- **Problem**: Only grafted on leaf nodes (almost never exists at depth 3-5)
- **Fix**: Graft onto any node at predicted depth

### Bug 5: Prefix Validation Added ✅
- **Problem**: No check if prefix reached target depth
- **Fix**: Validate `len(prefix_tokens) >= predicted_depth`, fallback if too short

---

## Configuration

### Default Parameters
```python
boundary_graft_threshold = 1.5  # Confidence threshold (0-1, lower = trigger SAM)
boundary_graft_max_sam_nodes = 8  # Max SAM tokens to add
boundary_graft_min_depth = 3  # Min depth for prediction
boundary_graft_max_depth = 8  # Max depth for prediction
```

### Threshold Semantics
- `threshold = 0.2` → Very selective (only trigger on very low confidence)
- `threshold = 0.5` → Moderate
- `threshold = 1.5` → Aggressive (always trigger, exp(logprob) < 1.5 for all negative logprobs)

**Note**: Default `1.5` is intentionally high for testing. Sweep will find optimal value.

---

## Testing Plan

### Phase 1: Sanity Checks (Manual)
```bash
# Quick syntax check
python -m py_compile samd/fusion/boundary_graft.py

# Import check
python -c "from samd.fusion.boundary_graft import predict_rejection_depth; print('OK')"

# Config validation
python -c "from samd.samd_config import SamdConfig; cfg = SamdConfig(fusion_mode='boundary_graft'); print('OK')"
```

### Phase 2: Dev Set (20 samples, 6 thresholds)
```bash
bash scripts/eval_boundary_graft_sweep.sh
```

Expected runtime: ~10-15 minutes
Output: `outputs/phase2_boundary_graft/humaneval_dev20_t{threshold}_n8.jsonl`

### Phase 3: Full Evaluation (164 samples, best threshold)
```bash
bash scripts/eval_boundary_graft.sh <best_threshold> 8 3 8 true
```

Expected runtime: ~6 minutes
Output: `outputs/phase2_boundary_graft/humaneval_full_t{threshold}_n8.jsonl`

---

## Success Criteria

From design doc:
- **MAT > 7.96 (+5%)** → SUCCESS, proceed to paper
- **MAT 7.58-7.96 (0-5%)** → PARTIAL, tune threshold
- **MAT < 7.58 (negative)** → FAILURE, diagnose

Baseline: `eagle_only` MAT = 7.58
Oracle ceiling: MAT = 8.31 (+9.65%)

---

## Known Limitations

1. **Single-path grafting**: Currently grafts onto first found path at predicted depth
   - Could extend to multi-path (graft on all leaves at depth)

2. **Threshold tuning required**: Default `1.5` is placeholder
   - Sweep will find optimal value empirically

3. **No SAM match-length gating**: Unlike Phase 1, doesn't check if SAM has good match
   - Could add `if sam_match_length < 5: skip_sam`

---

## Next Steps

1. **Run sanity checks** to verify no import/syntax errors
2. **Run threshold sweep** on dev set (q0-19)
3. **Analyze results**: MAT, prediction rate, SAM trigger rate
4. **Full evaluation** with best threshold
5. **Decision**:
   - Success → MT-Bench validation → paper
   - Partial → Add match-length gate, adaptive threshold
   - Failure → Negative result analysis

---

## Diff Summary

```
New files:
  samd/fusion/boundary_graft.py          (141 lines)
  scripts/eval_boundary_graft.sh         (48 lines)
  scripts/eval_boundary_graft_sweep.sh   (35 lines)

Modified files:
  samd/samd_config.py                    (+5 config fields, +1 validation)
  samd/utils.py                          (+183 lines for boundary_graft mode)
  evaluation/inference_samd.py           (+4 CLI args, +4 config params)

Total: +411 lines code
```

---

**Implementation Status**: ✅ Complete
**Review Status**: ✅ Gemini reviewed, critical bugs fixed
**Testing Status**: ⏳ Pending threshold sweep
