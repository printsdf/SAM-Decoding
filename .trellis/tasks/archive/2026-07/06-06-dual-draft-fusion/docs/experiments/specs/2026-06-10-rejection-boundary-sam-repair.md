# Experiment Design: Rejection-Boundary SAM Repair

**Date**: 2026-06-10
**Status**: Design phase
**Owner**: Qixuan Fu

---

## Goal

Implement and validate **Rejection-Boundary SAM Repair**: activate SAM candidates only at EAGLE3's predicted rejection boundary, achieving 5-7% real MAT improvement (target 50-70% of +9.65% oracle ceiling) on HumanEval.

## Hypothesis

**If** we predict EAGLE3's first rejection depth using draft confidence signals and insert SAM candidates only at that point, **then** we should achieve 5-7% MAT improvement over eagle3-only baseline **because** SAM provides exact retrieval-based repair where neural draft loses traction, with minimal overhead.

Oracle analysis confirms 100% SAM rescue rate at rejection boundaries (432/432 steps).

## Baseline to Beat

| Method | MAT | TPS | Notes |
|--------|-----|-----|-------|
| **eagle3_only** | 3.6689 | baseline | Primary baseline |
| sam_sequence_graft | 3.6812 (+0.33%) | ? | Existing SAM method (weak) |
| naive_fusion | ? | 0.816x | Negative baseline (Phase 2) |

**Success threshold**: MAT > 3.85 (+5% over eagle3_only) with TPS ≥ eagle3_only * 0.95

## Exact Metric and Selection Rule

**Primary metric**: Mean Accepted Tokens (MAT) on HumanEval q0-164

**Selection rule**:
- MAT improvement > 5% AND TPS degradation < 5% → SUCCESS, proceed to paper
- MAT improvement 3-5% → MARGINAL, need ablation to justify overhead
- MAT improvement < 3% → FAILURE, investigate gap between oracle and implementation

**Secondary metrics**:
- Tokens per second (TPS)
- SAM activation rate (% of steps where SAM is triggered)
- Rejection prediction accuracy (vs oracle ground truth)

## Dataset and Split

**Benchmark**: HumanEval (q0-164, full set)

**Why HumanEval**:
- Oracle gap +9.65% (vs MT-Bench +3.01%)
- Code generation has structural repetition (boilerplate, imports)
- EAGLE rejection rate 6.2% (vs MT-Bench 2.4%)

**No train/val split needed**: inference-only method, no learned parameters.

## Minimal Code Changes Expected

**Files to modify**:
1. `samd/fusion/rejection_boundary.py` (NEW) — core rejection-boundary logic
2. `samd/utils.py::gen_candidates()` — add `fusion_mode="rejection_boundary"` branch
3. `samd/samd_config.py` — add rejection threshold config
4. `evaluation/eval_rejection_boundary.py` (NEW) — evaluation script

**Estimated LOC**: 300-400 lines

## Implementation Approach

### Approach 1: Confidence-Based Threshold (Recommended)

**Mechanism**:
```python
# In gen_candidates()
if samd_config.fusion_mode == "rejection_boundary":
    eagle_confidence = compute_confidence(eagle_logprobs, depth)
    if eagle_confidence < REJECTION_THRESHOLD:
        # Activate SAM at predicted rejection depth
        sam_candidates = draft_sam_from_prefix(prefix, depth)
        candidates = merge_at_boundary(eagle_tree, sam_candidates, depth)
```

**Confidence signal**: Use EAGLE3 logprob at each depth. Low confidence → high rejection risk.

**Pros**: Simple, interpretable, no learned components
**Cons**: Threshold needs tuning (oracle analysis suggests optimal range)

### Approach 2: Entropy-Based Triggering

**Mechanism**: Monitor token-level entropy. High entropy → uncertain → trigger SAM.

**Pros**: Theoretically grounded
**Cons**: Adds compute overhead (entropy calculation per token)

### Approach 3: Fixed-Depth Heuristic

**Mechanism**: Always trigger SAM at depth=4 (oracle shows this is common rejection point).

**Pros**: Zero overhead
**Cons**: Not adaptive, misses dynamic rejection patterns

**Recommendation**: Start with Approach 1 (confidence threshold). It's the direct translation of oracle analysis and requires minimal new infrastructure.

## Sanity Checks Before Full Eval

### Checkpoint 1: Confidence Signal Validity (1 hour)
- Run eagle3-only on 10 HumanEval samples
- Log EAGLE confidence at each depth + actual rejection depth
- Verify correlation: low confidence → early rejection

**Pass criteria**: Spearman correlation > 0.5

### Checkpoint 2: SAM Activation Smoke Test (1 hour)
- Implement rejection-boundary mode
- Run on 10 samples
- Verify SAM is triggered at reasonable rate (target: 5-10% of steps)

**Pass criteria**: No crashes, SAM activated 5-20% of steps

### Checkpoint 3: MAT Sanity on Small Set (2 hours)
- Run full method on 20 HumanEval samples
- Compare MAT vs eagle3-only

**Pass criteria**: MAT improvement > 0% (any positive signal)

## Full Experiment Plan

### Phase 1: Threshold Tuning (4 hours)
- Grid search confidence thresholds: [0.3, 0.4, 0.5, 0.6, 0.7]
- Evaluate on HumanEval q0-50 (subset)
- Select threshold with best MAT

### Phase 2: Full Evaluation (8 hours)
- Run best threshold on HumanEval q0-164 (full)
- Measure MAT, TPS, activation rate
- Compare vs baselines (eagle3-only, sam_sequence_graft)

### Phase 3: Ablation (4 hours, if Phase 2 succeeds)
- Ablate SAM depth filter (test high-precision threshold from oracle)
- Ablate confidence signal (compare vs fixed-depth, entropy-based)

**Total compute budget**: ~16 GPU hours (1 A100 for 2 days)

## Failure Modes and Invalidation Evidence

| Failure Mode | Evidence | Next Action |
|--------------|----------|-------------|
| Confidence signal doesn't predict rejection | Checkpoint 1 fails (correlation < 0.3) | Switch to Approach 3 (fixed-depth) or abandon |
| SAM overhead dominates | TPS drops > 10% | Optimize SAM query or reduce activation rate |
| Implementation gap too large | MAT < +3% after Phase 2 | Deep dive: log actual vs predicted rejection points, check SAM match quality |
| Works only on subset | Good on q0-50, degrades on q0-164 | Investigate dataset variance, may need adaptive threshold |

**Invalidation threshold**: If Phase 2 MAT < +1%, the oracle-to-implementation gap is unsalvageable. Possible causes:
1. EAGLE confidence is noisy (doesn't reliably predict rejection)
2. SAM matches are too short/low-quality in practice
3. Verification overhead exceeds SAM benefit

## Next Decision After First Run

**If SUCCESS (MAT > +5%)**:
- Run on MT-Bench to verify generalization
- Write paper draft
- Prepare ablation studies for submission

**If MARGINAL (+3% < MAT < +5%)**:
- Profile overhead sources (SAM query time, merge logic)
- Try high-precision SAM filter (oracle showed +12.15% with threshold=5)
- Consider hybrid: rejection-boundary + high-precision

**If FAILURE (MAT < +3%)**:
- Analyze prediction accuracy: plot predicted vs actual rejection depths
- Check if SAM is being used: log SAM candidate counts and acceptance rates
- Consider pivoting to Structural Macro-Grafting (Gemini's recommendation)

## Risks and Mitigations

**Risk 1**: EAGLE confidence doesn't generalize from oracle analysis
**Mitigation**: Validate with Checkpoint 1 before full implementation

**Risk 2**: SAM query overhead hurts latency
**Mitigation**: Measure activation rate; target < 10% of steps

**Risk 3**: Confidence threshold is dataset-specific
**Mitigation**: Include adaptive threshold tuning in Phase 1

**Risk 4**: Oracle used ground-truth rejection, but we must predict it
**Mitigation**: Accept that implementation will be 50-70% of oracle ceiling. Target is still meaningful (+5-7% vs +9.65% oracle).

---

## Approval Checklist

Before proceeding to `experiment-planning`:

- [ ] Hypothesis clearly states expected outcome
- [ ] Baseline (eagle3-only, MAT=3.6689) is documented
- [ ] Success metric (MAT > 3.85, TPS > 95% baseline) is quantified
- [ ] Dataset (HumanEval q0-164) is specified
- [ ] Sanity checks (Checkpoints 1-3) are defined
- [ ] Failure modes have clear next actions
- [ ] Compute budget (16 GPU hours) is reasonable

**Ready for approval?** Yes — proceed to detailed experiment planning and implementation.
