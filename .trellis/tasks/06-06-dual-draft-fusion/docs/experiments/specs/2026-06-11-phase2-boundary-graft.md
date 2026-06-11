# Experiment Design: Phase 2 — Predicted-Boundary SAM Grafting

**Date**: 2026-06-11
**Status**: Design approved
**Owner**: Qixuan Fu

---

## Goal

Implement TRUE rejection-boundary SAM repair by predicting EAGLE's rejection depth and grafting SAM continuation at that point, achieving 5-7% MAT improvement on HumanEval with a single verification pass.

## Why Phase 1 Failed

Phase 1 implemented "root-level competition" (SAM vs EAGLE from root), not "sequential relay" (EAGLE prefix + SAM tail). Oracle proved relay value (+9.65%), but implementation did competition (上限 ~0.3% like sam_sequence_graft).

Root-level fusion cannot utilize EAGLE's accepted prefix — it's a different intervention than what the oracle measured.

## Hypothesis

**If** we predict EAGLE's likely rejection depth using per-node confidence margins, and graft SAM continuation candidates at that predicted depth into the EAGLE tree BEFORE verification, **then** MAT will improve by 5-7% over eagle-only on HumanEval, **because** the single verification pass will accept SAM's tail tokens where EAGLE would have been rejected.

## Approach: Per-Depth Margin Prediction + SAM Grafting

### Algorithm

```python
# Step 1: EAGLE drafting with logprobs
eagle_tokens, eagle_buffers, eagle_logprobs = eagle.gen_draft(start_token, return_logprobs=True)

# Step 2: Predict rejection depth from per-depth margins
predicted_depth = predict_rejection_depth(
    logprobs=eagle_logprobs,
    position_ids=eagle_buffers['tree_position_ids'],
    threshold=boundary_graft_threshold  # e.g., margin < 1.5 at depth
)

if predicted_depth > 0 and predicted_depth < max_depth:
    # Step 3: Get EAGLE prefix up to predicted depth
    prefix_tokens = extract_prefix(eagle_tree, predicted_depth)

    # Step 4: SAM continuation from that prefix
    sam_continuation = sam.gen_draft_from_prefix(
        prefix_tokens,
        max_len=boundary_graft_max_sam_nodes  # 4-8 nodes
    )

    # Step 5: Graft SAM tail onto EAGLE leaves at predicted depth
    fused_tree = graft_sam_at_depth(
        eagle_tree=eagle_tree,
        sam_candidates=sam_continuation,
        graft_depth=predicted_depth,
        max_added_nodes=boundary_graft_max_sam_nodes
    )
else:
    fused_tree = eagle_tree

# Step 6: Single verification pass
accepted = verify(fused_tree)
```

### Key differences from Phase 1

| Aspect | Phase 1 (failed) | Phase 2 (this) |
|--------|------------------|----------------|
| SAM起点 | Root (depth 0) | EAGLE accepted prefix (depth D) |
| 目标 | 替代 EAGLE | 补全 EAGLE tail |
| Verify次数 | 1次 | 1次 |
| Oracle对应 | ❌ root竞争 | ✅ boundary接力 |

## Baseline to Beat

| Method | MAT | Status |
|--------|-----|--------|
| eagle_only | 7.58 | HumanEval full (current) |
| sam_sequence_graft | ~7.60 (+0.3%) | Oracle estimate |
| Phase 1 (root-level) | 6.73 (-11%) | Failed |
| **Oracle ceiling** | **8.31 (+9.65%)** | Target |

## Success Metric

```
MAT > 7.96 (+5%)       →  SUCCESS, proceed to paper
MAT 7.58-7.96 (0-5%)   →  PARTIAL, tune threshold
MAT < 7.58 (negative)  →  FAILURE, diagnose
```

## Dataset and Split

- **Primary**: HumanEval q0-164 (full)
- **Secondary**: MT-Bench q0-80 (cross-validation, expect smaller gain ~3%)

## Code Changes

### 1. New: `samd/fusion/boundary_graft.py` (~200 lines)

```python
def predict_rejection_depth(
    logprobs: List[float],
    position_ids: torch.Tensor,
    threshold: float
) -> int:
    """Predict first depth where EAGLE will likely be rejected.

    Returns depth D where avg margin at depth D < threshold.
    If no such depth, returns -1 (no prediction).
    """
    pass

def graft_sam_at_depth(
    eagle_tree: TreeSpec,
    sam_candidates: List[int],
    graft_depth: int,
    max_added_nodes: int
) -> TreeSpec:
    """Graft SAM continuation onto EAGLE leaves at graft_depth.

    Preserves EAGLE nodes at depth <= graft_depth.
    Adds SAM nodes as children of EAGLE leaves.
    """
    pass
```

### 2. Update: `samd/utils.py::gen_candidates()`

Add `fusion_mode="boundary_graft"` branch (~100 lines).

### 3. Update: `samd/samd_config.py`

```python
fusion_mode: Literal[..., "boundary_graft"] = ...
boundary_graft_threshold: float = 1.5  # Margin threshold for prediction
boundary_graft_max_sam_nodes: int = 8  # SAM tail length
```

## Sanity Checks

### Check 1: Prediction distribution (1 hour)
```python
# Log predicted depths on 20 samples
python eval_log_predictions.py --samples 20
# Expected: depths concentrated in 4-7 range
```

**Pass criteria**: 80% of predictions in [3, 8]

### Check 2: SAM transfer success rate (1 hour)
```python
# Log SAM transfer_state() success on 20 samples
# Expected: > 50% can match EAGLE prefix
```

**Pass criteria**: Transfer success > 40%

### Check 3: TreeSpec validity (1 hour)
```python
# Run 20 samples, check no crashes
# Verify tree_attn_mask shape matches tokens
```

**Pass criteria**: No crashes, valid tree shapes

## Full Experiment Plan

### Threshold sweep (6 runs, 36 minutes)

```bash
for threshold in 0.5 1.0 1.5 2.0 2.5 3.0; do
    bash eval_boundary_graft.sh --threshold $threshold
done
```

**Select**: threshold with best MAT

### Full evaluation (1 run, 6 minutes)

```bash
bash eval_boundary_graft.sh --threshold <best> --full
```

**Measure**: MAT, SAM trigger rate, average graft depth

## Failure Modes

| Failure | Evidence | Root Cause | Next Action |
|---------|----------|------------|-------------|
| Prediction too shallow | MAT drops, avg predicted_depth < 3 | Threshold too high | Lower threshold |
| Prediction too deep | No effect, avg predicted_depth > 7 | Threshold too low | Raise threshold |
| SAM transfer fails | Transfer rate < 20% | Prefix mismatch | Add fallback or relax match |
| TreeSpec crashes | Verification errors | Graft logic bug | Debug tree construction |
| No improvement across all thresholds | MAT ≤ 7.58 for all | SAM tail无匹配 or prediction无关 | Pivot to negative result |

## Invalidation Threshold

If MAT < 7.58 across threshold range [0.5, 3.0], then:
- SAM cannot continue from EAGLE prefix (index coverage issue)
- OR prediction has no correlation with actual rejection
- **Action**: Abandon this direction, write negative result analyzing why oracle-to-implementation gap exists

## Compute Budget

- Sanity checks: 3 hours
- Threshold sweep: 36 minutes (6 runs)
- Full evaluation: 6 minutes
- **Total**: ~4 hours GPU time

## Next Decision

**If MAT > 7.96**:
- Run MT-Bench cross-validation
- Lock configuration
- Write paper draft

**If MAT 7.58-7.96**:
- Add SAM match_length gate (only graft if match > 5)
- Try adaptive threshold per step
- Profile: which steps benefit vs hurt

**If MAT < 7.58**:
- Plot predicted vs actual rejection correlation
- Analyze why SAM tail fails to help
- Consider Structural Macro-Grafting or close direction

---

## Approval

- [x] Hypothesis明确
- [x] Baseline明确 (eagle_only = 7.58)
- [x] Success metric量化 (> 7.96 = +5%)
- [x] Dataset固定 (HumanEval q0-164)
- [x] Failure modes with next actions
- [x] 单次 verify 约束满足

**Ready for implementation**: YES
