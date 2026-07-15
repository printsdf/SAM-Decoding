# Experiment Design: Boundary Predictor Calibration

**Date**: 2026-06-11
**Status**: Approved for experiment planning; implementation not started
**Owner**: Qixuan Fu

---

## Goal

Before implementing online SAM grafting, test whether EAGLE3 confidence signals
can predict the concrete node/path where target verification first rejects the
EAGLE tree.

The experiment answers one question:

> Is there enough node-level rejection-boundary signal to justify an online
> Rejection-Boundary SAM Repair implementation?

This supersedes direct depth-only boundary grafting as the next step. Depth-only
prediction and probability thresholds above 1 are now treated as baselines or
failure cases, not as the main method.

## Hypothesis

If low node-level confidence in the EAGLE tree predicts target-model rejection,
then a margin/logprob risk score should identify true rejection boundaries at a
much higher precision than random or depth-prior baselines, because SAM's
HumanEval oracle value is concentrated at those EAGLE failure points.

## Baseline to Beat

Known HumanEval oracle numbers:

| Method | MAT | Gap vs EAGLE-only | Notes |
| --- | ---: | ---: | --- |
| EAGLE3-only | 3.6689 | +0.00% | Primary baseline |
| `sam_sequence_graft` | 3.6812 | +0.33% | Existing weak SAM method |
| perfect oracle | 4.0231 | +9.65% | Full same-trace ceiling |
| rejection-boundary oracle | 4.0231 | +9.65% | All SAM value at EAGLE rejection points |

Predictor baselines:

| Predictor | Meaning |
| --- | --- |
| Random triggered node | Controls for trigger rate and tree size |
| Depth prior | Always choose the empirically most common rejection depth |
| Depth-only low-confidence | Choose a depth, then first path at that depth |
| High-margin trigger | Tests the "confidently wrong" alternative |
| Low-margin node trigger | Main interpretable hypothesis |
| Risk-score node trigger | Margin + local logprob + path logprob |

## Dataset and Split

Primary benchmark:

```text
HumanEval q0-164
```

Calibration split:

```text
q0-82    threshold/model-selection split
q82-164  held-out validation split
```

No learned model is required for the first pass. If a learned predictor is later
tested, it must use the same split and report held-out metrics.

MT-Bench and MedQA are not part of this first experiment because prior oracle
ceilings were below the Phase B gate.

## Required Trace Schema

Each decode step must include:

- EAGLE tree nodes:
  - `tree_index`
  - `parent_index`
  - `depth`
  - `token`
  - `path_tokens`
  - `local_logprob`
  - `rank_among_siblings`
  - `sibling_margin`
  - `cumulative_path_logprob`
- verifier labels:
  - `acceptance_path`
  - `first_rejected_depth`
  - `first_rejected_tree_index`, when the rejected EAGLE node is identifiable
- SAM evidence:
  - SAM candidate path or sequence
  - SAM match length
  - whether SAM has an accepted continuation from the true rejection boundary

Existing fusion-profile traces are useful but may not contain all node-level
features. If the schema is incomplete, the first implementation task should be a
trace-enrichment pass, not online grafting.

## Candidate Approaches

### Approach A: Depth-Only Predictor

Predict a rejection depth from aggregate per-depth confidence and graft on the
first path at that depth.

Decision: baseline only.

Reason: this can graft SAM onto the wrong branch even when the depth is correct.

### Approach B: Node-Level Low-Margin Predictor

For nodes in verifier order:

```text
trigger if depth >= min_depth and sibling_margin < margin_threshold
```

Decision: recommended first test.

Reason: margin semantics are clear: larger margin means more confidence, smaller
margin means uncertainty. This directly tests the current low-confidence repair
hypothesis.

### Approach C: Node-Level Risk Score

Score each node:

```text
risk(node) =
    w_margin * (-sibling_margin)
  + w_logprob * (-local_logprob)
  + w_path * (-(cumulative_path_logprob / depth))
  + w_depth * depth_penalty
```

Decision: second test after Approach B.

Reason: it may capture rejection risk better than margin alone, while remaining
interpretable and non-learned.

## First Experiment

Smallest falsifiable test:

1. Generate or load HumanEval fusion-profile traces with enriched EAGLE node
   features.
2. Reconstruct true first EAGLE rejection boundary per decode step.
3. Sweep low-margin thresholds on `q0-82`.
4. Evaluate the selected threshold on `q82-164`.
5. Compare against random, depth-prior, depth-only, and high-margin baselines.

No online generation behavior should change in this experiment.

## Metrics and Selection Rule

### Signal Metrics

| Metric | Definition | Pass criterion |
| --- | --- | --- |
| trigger rate | triggered steps / all steps | <= 15% |
| boundary precision | triggered true boundaries / triggered steps | >= 20% |
| boundary recall | triggered true boundaries / all true rejection steps | >= 35% |
| precision lift | precision / rejection-step base rate | >= 3x |
| median depth error | median `abs(predicted_depth - true_depth)` on true positives | <= 1 |
| path-prefix hit rate | predicted path shares true rejected prefix | >= 50% on true positives |

HumanEval base rejection rate is about 6.2%, so 20% precision is already a
meaningful lift over random triggering.

### Oracle Utility Metrics

| Metric | Definition | Pass criterion |
| --- | --- | --- |
| predicted-boundary oracle MAT | MAT if SAM rescue is allowed only at predicted boundaries | >= EAGLE-only * 1.03 |
| recovered oracle ceiling | `(predicted_mat - eagle_mat) / (perfect_mat - eagle_mat)` | >= 30% |
| low-margin vs high-margin | low-margin oracle gap beats high-margin oracle gap | required |
| node-level vs depth-only | node-level oracle gap beats depth-only oracle gap | required |

Proceed to online implementation only if both signal metrics and oracle utility
metrics pass on the held-out split.

## Sanity Checks

Before trusting results:

1. Recomputed EAGLE-only MAT from trace must match `3.6689 +/- 0.02` on full
   HumanEval or the split-equivalent value on subsets.
2. Recomputed rejection-boundary oracle on the full trace should recover about
   `+9.65%` gap and `432 / 6932` rejection steps.
3. Thresholds for `exp(logprob)` scores must stay in `[0, 1]`.
4. Margin-based thresholds must preserve direction:

   ```text
   low margin -> more uncertain -> more likely to trigger
   ```

5. Trace-on/profile outputs are diagnostic only; no throughput claim is made
   from this experiment.

## Expected Code or Config Changes

This is a design spec, not an implementation plan. Expected later changes are:

- add or extend a trace-enrichment path that emits node-level EAGLE features;
- add an offline analyzer for boundary-predictor sweeps;
- add tests for margin direction, threshold scale, and depth-vs-node predictor
  behavior.

Do not add online `fusion_mode="boundary_graft"` behavior until this calibration
experiment passes.

## Compute Budget

Local:

- Document and analyzer review only.

Remote model host:

- 20-sample enriched trace smoke: up to 1 GPU hour.
- Full HumanEval q0-164 enriched trace/profile: up to 4 GPU hours.
- Offline sweeps: CPU-only, minutes.

Total first-pass budget: <= 5 GPU hours.

## Failure Modes and Decisions

| Failure | Evidence | Decision |
| --- | --- | --- |
| No predictive signal | Precision lift < 2x or recall < 20% | Do not implement margin-gated online grafting |
| Wrong direction | High-margin trigger beats low-margin trigger | Reframe as high-confidence-error detector and redesign |
| Depth-only equals node-level | Node-level does not beat depth-only | Simpler topology/branch model may be sufficient; inspect paths |
| SAM cannot rescue predicted boundaries | Oracle MAT gap < +3% | Pivot away from boundary repair or add high-precision SAM gate |
| Trace schema insufficient | Cannot identify true rejected path/node | Implement trace enrichment before any online experiment |

## Next Decision

If the held-out validation split passes:

- move to `experiment-planning` for an online node-level boundary graft
  implementation;
- use the selected threshold/risk score only as a frozen configuration for the
  first online run.

If it fails:

- write a negative result explaining whether the failure is predictor signal,
  SAM rescue quality, or trace/schema limitation;
- consider high-precision SAM or structural macro-grafting instead.

## Approval Checklist

- [x] Hypothesis is one sentence and falsifiable.
- [x] Baselines include EAGLE-only, weak SAM graft, random, depth-only, and
  high-margin controls.
- [x] Metrics and selection rule are exact.
- [x] Dataset and split are fixed.
- [x] Compute budget is bounded.
- [x] First experiment is offline and can disprove the idea before online code.

**Ready for user approval**: pending.
