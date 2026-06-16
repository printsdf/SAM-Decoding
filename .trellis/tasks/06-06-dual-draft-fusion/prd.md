# Dual Draft Fusion PRD

Status: in progress — Phase B Drafter-MARS q0-20 smoke failed low-ceiling gate
Last updated: 2026-06-16
Task: `dual-draft-fusion`

## Problem

Naive SAM + EAGLE3 candidate fusion and depth-decoupled SAM tail extension did
not produce enough oracle upside on dialogue tasks. However, on code generation
(HumanEval), SAM shows strong oracle value specifically at EAGLE rejection
boundaries.

The validated research direction is **Rejection-Boundary SAM Repair**: use SAM
only where EAGLE has already failed, not as a broad second draft source.

## Closed Decisions

| Direction | Decision | Evidence |
| --- | --- | --- |
| Naive fusion with depth proxy | Negative baseline | Lower MAT/TPS than EAGLE3-only |
| Naive fusion with real EAGLE3 logprobs | Negative/neutral baseline | MT-Bench `0.962x` EAGLE3 TPS |
| Depth-decoupled fusion on MedQA | Reject | Best oracle gap `+0.00%` |
| Depth-decoupled fusion on MT-Bench | Reject | Best oracle gap `+0.61%`, below 3% gate |
| Rejection-Boundary on MT-Bench | Weak positive | Oracle gap `+3.01%` (= perfect ceiling) |
| High-Precision SAM on MT-Bench | Below gate | Best gap `+3.42%` < 8% gate |
| Drafter-MARS SAM gate on HumanEval q0-20 | Failed smoke / low-ceiling debug | Held-out predicted MAT gain `+0.83%`; same-trace perfect/rejection-boundary gap only `+3.91%` overall and `+3.38%` held-out |

## Validated Direction: Rejection-Boundary SAM Repair

### Oracle Evidence (HumanEval)

| Metric | Value | Gate | Status |
| --- | --- | --- | --- |
| Perfect oracle gap | +9.65% | > 5% | PASS |
| Rejection-boundary oracle gap | +9.65% | > 7% | PASS |
| High-precision SAM gap (threshold=5) | +12.15% | > 8% | PASS |
| EAGLE rejection steps | 6.2% of steps | - | - |
| SAM rescue rate | 100% | - | - |

### Key Insight

Rejection-boundary oracle gap equals the perfect oracle ceiling (both +9.65%).
This means ALL of SAM's value on HumanEval is concentrated at EAGLE rejection
points. The implementation strategy is therefore clear: activate SAM only at
predicted rejection boundaries.

### Dataset Comparison

| Dataset | Perfect ceiling | Rejection-boundary | EAGLE rejection % |
| --- | --- | --- | --- |
| MedQA | +1.45% | +3.01% (approx) | 2.4% |
| MT-Bench | +3.01% | +3.01% | 2.4% |
| HumanEval | +9.65% | +9.65% | 6.2% |

Code generation has 3x the oracle ceiling of dialogue tasks.

## Baselines

| Baseline | Purpose |
| --- | --- |
| `eagle_only` | Primary MAT baseline (HumanEval MAT=3.67) |
| `sam_sequence_graft` | Existing SAM insertion (HumanEval +0.33%) |
| `perfect` oracle | Ceiling (+9.65% on HumanEval) |

## Phase B: Implementation Plan

Target: implement rejection-boundary SAM repair and measure real speedup, but
only after an offline predictor gate passes.

Current sequence:

1. Calibrate a sparse rejection-boundary predictor from HumanEval profile
   traces.
2. Drafter-MARS style EAGLE3 raw-logit top-1/top-2 ratios were tested on a
   q0-20 smoke, but the held-out predicted MAT gain failed the `+2%` smoke gate.
3. Do not run full HumanEval q0-164 calibration for the Drafter-MARS ratio gate
   until the q0-20 low ceiling is explained or a new smoke trace shows enough
   same-trace oracle headroom.
4. Only after a future offline gate passes, implement online SAM boundary
   grafting.
5. Measure trace-off TPS on HumanEval only after an online implementation exists
   (target: `> eagle_only * 1.05`).

## Current Status

The previous unconditioned low-margin boundary-predictor smoke is closed as a
diagnostic failure. It improved oracle MAT on q10-20 but triggered on `99.89%`
of held-out decode steps, which degenerates into always-on SAM repair.

The Drafter-MARS SAM gate smoke failed the current q0-20 gate:

```text
Drafter-MARS SAM gate
HumanEval q0-20 smoke
train: q0-10
heldout: q10-20
predictor-suite: drafter_mars
selected_method: draft_mars_reachable
heldout_trigger_rate: 13.21%
heldout_boundary_precision: 21.43%
heldout_eagle_mat: 7.0811
heldout_predicted_boundary_oracle_mat: 7.1396
predicted MAT gain: +0.83%
required MAT gain: >= +2.00%
pass: false
```

This experiment used EAGLE3 drafter raw logits, not target logits and not the
old unconditioned sibling margin. It triggered sparsely enough and met the
precision gate, but the MAT gain was too small. More importantly, the same
trace reported a perfect oracle gap of only `+3.38%`, far below the prior
HumanEval rejection-boundary ceiling of `+9.65%`. After syncing the missing
auxiliary oracle scripts, the same q0-20 trace produced a standalone
rejection-boundary oracle with `75/75` SAM rescues and an overall gap of
`+3.91%`, equal to the same-trace perfect oracle. The low ceiling is therefore
real for this trace/split; the q0-20 smoke should not be escalated to full
calibration.

## Non-Goals

- Do not target MT-Bench or MedQA for this direction (ceiling too low).
- Do not build always-on fusion; SAM activates only at rejection boundaries.
- Do not build Static SAM from evaluation answers.
- Do not compare throughput across machines or trace settings.

## Next Step

Do not run full q0-164 calibration or online grafting for Drafter-MARS SAM
gating from this q0-20 result. First compare the q0-20 provenance and trace
statistics against the prior HumanEval oracle trace and explain why EAGLE-only
MAT is much higher and the same-trace oracle ceiling is much lower. If Phase B
continues after that, write a new or amended experiment spec/plan before any
model run.

Closed result:

```text
.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/2026-06-11-drafter-mars-sam-gate.md
```

Do not implement online `fusion_mode="boundary_graft"` or report throughput
speedup until a future offline predictor gate passes.
