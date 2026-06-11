# Experiment Design: Drafter-MARS SAM Gate

**Date**: 2026-06-11
**Status**: Proposed; awaiting experiment planning
**Owner**: Qixuan Fu

---

## Goal

Test whether a MARS-style adaptive margin computed from EAGLE3 drafter logits
can decide when to include SAM repair candidates before target verification.

This replaces the failed unconditioned sibling-margin trigger from
`results/2026-06-11-boundary-predictor-calibration.md`.

The experiment answers one question:

> Can drafter-side top-1/top-2 logit closeness identify reachable EAGLE
> uncertainty points where SAM repair recovers target acceptance, without
> triggering on nearly every decode step?

## Hypothesis

If EAGLE3 is uncertain on the top-path or high-reachability parent, then a
MARS-style drafter logit ratio should trigger SAM on a small fraction of decode
steps while recovering a meaningful fraction of the rejection-boundary oracle
ceiling, because the uncertainty is measured at parents the verifier is likely
to reach rather than anywhere in the EAGLE tree.

## Baseline to Beat

HumanEval q0-20 smoke from the previous boundary-predictor calibration:

| Method | Held-out q10-20 MAT | Notes |
| --- | ---: | --- |
| EAGLE3-only | 3.9833 | Same trace, `node_budget=60` |
| perfect oracle | 4.3449 | Same-split upper bound |
| unconditioned low-margin trigger | 4.1406 | Trigger rate `99.89%`, diagnostic only |

Full HumanEval reference from prior oracle work:

| Method | MAT | Gap vs EAGLE-only | Notes |
| --- | ---: | ---: | --- |
| EAGLE3-only | 3.6689 | +0.00% | Primary full-run baseline |
| `sam_sequence_graft` | 3.6812 | +0.33% | Existing weak SAM method |
| rejection-boundary oracle | 4.0231 | +9.65% | Full same-trace ceiling |

The new method must beat EAGLE3-only under the same node budget and must not
repeat the previous always-on trigger failure.

## Method Definition

Use EAGLE3 drafter raw logits, not target logits:

```text
z1(parent) = drafter top-1 child raw logit under this parent
z2(parent) = drafter top-2 child raw logit under this parent
ratio(parent) = z2(parent) / z1(parent)
```

MARS-adaptive trigger:

```text
trigger(parent) = ratio(parent) > theta
```

Equivalent adaptive margin form:

```text
z1(parent) - z2(parent) < (1 - theta) * z1(parent)
```

This is a drafter-side proxy, not MARS itself. MARS uses target logits during
verification; this experiment intentionally uses drafter logits to avoid a
second target verification pass.

Do not use `sibling_margin` from the previous analyzer as the primary feature.
If raw drafter logits are unavailable, use fixed logprob delta only as a
fallback control:

```text
logprob_top1(parent) - logprob_top2(parent) < tau
```

## Candidate Approaches

### Approach A: `draft_mars_top_path`

Only evaluate parents on the EAGLE greedy top path. A parent is on the top path
if every child chosen so far has `rank_among_siblings == 1`.

Trigger on the first top-path parent whose drafter logit ratio satisfies:

```text
z2 / z1 > theta
```

Recommendation: run first.

Reason: it is the smallest test of the new idea and directly prevents the
previous "any branch in the tree" always-trigger failure.

### Approach B: `draft_mars_reachable`

Evaluate parents whose prefix is likely under the drafter:

```text
normalized_path_logprob = cumulative_path_logprob / depth
```

Select a train-calibrated reachability quantile, then trigger on the earliest
reachable parent satisfying:

```text
z2 / z1 > theta
```

Recommendation: second ablation.

Reason: this can recover non-greedy but still likely branches, at the cost of
one extra hyperparameter.

### Approach C: `draft_delta_top_path`

Fallback when raw drafter logits are missing or top-1 logits are not positive:

```text
logprob_top1 - logprob_top2 < tau
```

Recommendation: control only.

Reason: fixed deltas are model-scale dependent and do not preserve the adaptive
margin property from MARS.

## Dataset and Split

Smoke test:

```text
HumanEval q0-20
train: q0-10
heldout: q10-20
```

Full calibration only if smoke passes:

```text
HumanEval q0-164
train: q0-82
heldout: q82-164
```

Use the same trace/profile settings as boundary-predictor calibration:

```text
tree_method=eagle3
tree_fusion=none
fusion_mode=naive
samd_n_predicts=40
samd_len_threshold=5
samd_len_bias=5
fusion_max_draft_tokens=60
node_budget=60
max_cache_len=4096
```

## Required Trace Schema

Each EAGLE parent considered by the analyzer needs:

```text
parent_tree_index
parent_token_path
parent_depth
parent_cumulative_path_logprob
child_top1_token
child_top2_token
child_top1_logit
child_top2_logit
child_top1_logprob
child_top2_logprob
child_top1_rank
child_top2_rank
```

Derived fields:

```text
draft_logit_ratio = child_top2_logit / child_top1_logit
draft_adaptive_margin = child_top1_logit - child_top2_logit
draft_adaptive_margin_threshold = (1 - theta) * child_top1_logit
is_top_path_parent
normalized_path_logprob
```

Verifier labels from the existing calibration remain required:

```text
acceptance_path
first_rejected_depth
first_rejected_parent_path
first_rejected_parent_index
```

SAM evidence remains trace-only oracle evidence:

```text
SAM candidate path matches acceptance suffix after predicted boundary
```

## Metrics and Selection Rule

Threshold grid:

```text
theta in {0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98}
```

For `draft_mars_reachable`, also sweep train quantiles:

```text
reachable_quantile in {0.50, 0.70, 0.85, 0.95}
```

Select thresholds on the train split only. Maximize train
`predicted_boundary_oracle_mat` subject to:

```text
trigger_rate <= 15%
boundary_precision >= 20%
```

Tie break by:

```text
higher recovered_oracle_ceiling
then higher boundary_recall
then lower trigger_rate
```

Held-out pass criteria:

| Metric | Smoke pass | Full pass |
| --- | ---: | ---: |
| trigger rate | <= 20% | <= 15% |
| boundary precision | >= 15% | >= 20% |
| predicted MAT vs EAGLE | >= +2% | >= +3% |
| recovered oracle ceiling | >= 20% | >= 30% |
| beats unconditioned trigger efficiency | required | required |

Efficiency comparison must report:

```text
MAT gain per triggered step
MAT gain per added SAM candidate
```

## Smallest Falsifiable Experiment

1. Add or confirm trace fields for drafter raw top-1/top-2 logits per EAGLE
   parent.
2. Run HumanEval q0-20 smoke with enriched trace.
3. Analyze `draft_mars_top_path`, `draft_mars_reachable`, and
   `draft_delta_top_path`.
4. Stop if all variants either trigger above 20% or fail to beat EAGLE by 2%
   on heldout q10-20.

Do not run full q0-164 until the smoke result avoids the previous always-on
failure.

## Sanity Checks

- Raw drafter logits must be used for the ratio. Do not compute a ratio from
  logprobs.
- Report the fraction of considered parents with `child_top1_logit <= 0`. If it
  is nontrivial, the ratio metric is unstable and the fixed-delta fallback
  becomes mandatory.
- Increasing `theta` should make the gate stricter and lower trigger rate.
- `draft_mars_top_path` must trigger less often than the old unconditioned
  low-margin predictor on the same trace.
- Same-budget oracle accounting must hold:

```text
eagle_mat <= predicted_boundary_oracle_mat <= perfect_mat
```

- No target logits may be used by the predictor.
- Trace-on/profile results are diagnostic only; do not report throughput
  speedup from this experiment.

## Expected Code or Config Changes

- Extend EAGLE trace enrichment to preserve raw top-1/top-2 child logits per
  parent when available.
- Add parent-level Drafter-MARS predictor variants to
  `evaluation/analyze_boundary_predictor.py`.
- Add tests for:
  - raw-logit ratio direction;
  - fixed-delta fallback when raw logits are missing;
  - top-path filtering prevents unrelated low-margin branches from triggering;
  - same-budget predicted MAT does not exceed perfect MAT.

No online `fusion_mode` change is allowed in this experiment.

## Compute Budget

Remote model host:

```text
q0-20 smoke: <= 1 GPU hour
q0-164 full run, only after smoke pass: <= 4 GPU hours
offline analysis: CPU-only
```

Local:

```text
syntax checks and synthetic analyzer tests only
```

## Failure Modes

| Failure | Evidence | Decision |
| --- | --- | --- |
| Drafter uncertainty is not target rejection signal | Precision near random and MAT gain < +2% | Do not use drafter margin as gate |
| Trigger still too broad | Heldout trigger rate > 20% in smoke | Add stricter reachability filter or reject |
| Ratio unstable | Many `child_top1_logit <= 0` cases | Use fixed-delta fallback only |
| Utility concentrated off top path | top-path fails but reachable succeeds | Continue with reachable parent gate |
| SAM cannot rescue triggered boundaries | Precision ok but MAT gain small | Pivot away from SAM repair for this gate |

## Next Decision

If smoke passes, move to `experiment-planning` for implementation/run steps and
then execute full HumanEval q0-164 calibration.

If smoke fails, record a negative result and do not implement online
Drafter-MARS SAM gating.
