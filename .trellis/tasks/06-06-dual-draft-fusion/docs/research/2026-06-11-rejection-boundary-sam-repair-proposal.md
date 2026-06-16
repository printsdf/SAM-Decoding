# Rejection-Boundary SAM Repair Proposal

## Date
2026-06-11

## Status

Proposal draft. This note records the research idea and semantic contracts before
experiment design or implementation. It should not be treated as validated code
design yet.

## Context

Previous fusion directions showed weak or negative real upside:

- Naive SAM + EAGLE3 fusion was negative or neutral.
- Depth-decoupled SAM tail extension did not create enough oracle gap on MT-Bench
  or MedQA.
- HumanEval is different: local oracle analysis shows SAM value concentrated at
  EAGLE rejection points.

The current working hypothesis is therefore not broad dual-draft fusion. It is:

> Use SAM as a local repair module only at predicted EAGLE rejection boundaries.

## Core Claim

EAGLE3 is usually the right drafter. SAM should not compete with it globally.
Instead, SAM should be activated only when EAGLE is likely to fail, and it should
be grafted onto the exact EAGLE path near the predicted failure point.

In paper language:

> Rejection-aware dual drafting treats non-parametric retrieval as a boundary
> repair mechanism, not as an always-on second draft source.

## Evidence So Far

From the active PRD:

| Metric | HumanEval value |
| --- | ---: |
| Perfect oracle gap | +9.65% |
| Rejection-boundary oracle gap | +9.65% |
| High-precision SAM gap, threshold=5 | +12.15% |
| EAGLE rejection steps | 6.2% |
| SAM rescue rate in oracle analysis | 100% |

Interpretation: on HumanEval, the rejection-boundary oracle equals the perfect
oracle ceiling. That suggests SAM's useful contribution is not distributed
across all decode steps; it is concentrated where EAGLE first fails.

## Margin and Threshold Semantics

This must be fixed before implementation.

If margin is defined as:

```text
margin = top1_logprob - top2_logprob
```

then the natural interpretation is:

```text
larger margin  -> EAGLE is more certain
smaller margin -> EAGLE is less certain
```

Therefore, a low-confidence trigger should normally use:

```text
trigger_sam = margin < margin_threshold
```

not:

```text
trigger_sam = margin > margin_threshold
```

If later experiments show "high-confidence wrong" is the useful slice, that
should be represented as a separate calibrated risk feature, not by redefining
the basic margin semantics.

For probability thresholds, keep the scale explicit. If a score is
`exp(logprob)`, it is in probability space and should usually be in `[0, 1]`.
Thresholds such as `1.5` are invalid for this scale and will make nearly every
node look below threshold.

Recommended naming:

| Concept | Meaning |
| --- | --- |
| `margin` | confidence gap, larger is more certain |
| `margin_threshold` | low-confidence cutoff, trigger when `margin < threshold` |
| `rejection_risk` | calibrated failure score, larger is more risky |
| `risk_threshold` | trigger when `rejection_risk > threshold` |

## Method Sketch

### Step 1: Generate an EAGLE Tree

For every non-root EAGLE node, record:

- `node_id`
- `parent_id`
- `depth`
- `token`
- `path_tokens`
- `local_logprob`
- `rank_among_siblings`
- `sibling_margin`
- `cumulative_path_logprob`

The key design point is that the predictor must operate on nodes or paths, not
only on depths. A low-confidence depth is not enough to identify where SAM should
be grafted.

### Step 2: Predict a Rejection Boundary

Minimum viable rule:

```text
for node in verifier_order:
    if node.depth >= min_depth and node.sibling_margin < margin_threshold:
        predicted_boundary = node
        break
```

Better non-learned score:

```text
rejection_risk(node) =
    w_margin * (-sibling_margin)
  + w_logprob * (-local_logprob)
  + w_path * (-(cumulative_path_logprob / depth))
  + w_depth * depth_penalty
```

The first experiments should sweep thresholds and feature combinations before
training any learned predictor.

### Step 3: Graft SAM at the Predicted Path

If the predicted rejected node is:

```text
root -> ... -> parent -> predicted_node
```

then SAM should be transferred to the prefix ending at `parent`, and its
continuation should be grafted as an alternative child/chain from that parent.

This is different from depth-only grafting:

```text
bad: choose first path at depth D
good: choose the concrete path that produced the rejection-risk signal
```

The target model verifier remains unchanged. The method only changes the
candidate tree exposed to standard speculative verification.

## Minimal Experiment Before Online Implementation

Before changing the online generation path, run offline calibration using
existing or newly captured profile traces.

Required labels per decode step:

- EAGLE candidate tree with node ids, parents, depth, path, and logprobs.
- Verifier accepted path and first rejected node/depth.
- SAM candidate sequence and match length.
- Whether SAM could rescue from the true rejection boundary.

Questions to answer:

1. Does margin or logprob predict first rejection better than random/depth-only?
2. Is the best trigger low-margin, high-margin, or a mixed risk score?
3. Does predicted-node grafting beat depth-only grafting?
4. What trigger rate is needed to get meaningful MAT gain?
5. Does SAM overhead erase the acceptance gain?

Suggested offline metrics:

| Metric | Purpose |
| --- | --- |
| boundary precision | How often a triggered boundary is near true rejection |
| boundary recall | How much of the true rejection opportunity is covered |
| depth error | How close predicted depth is to true first rejection depth |
| node/path hit rate | Whether predicted path matches the rejecting path |
| SAM rescue rate | Whether SAM adds an accepted continuation at boundary |
| trigger rate | Runtime overhead proxy |

## Online Success Criteria

Recommended gates for HumanEval:

- `TPS >= eagle_only * 1.05`
- `SAM trigger_rate < 15%`
- SAM grafted nodes have higher accepted-node rate than naive SAM fusion nodes.
- MAT improvement is not explained solely by larger tree size.
- Trace-off runs are used for speedup; trace/profile runs are used only for
  diagnosis.

## Required Ablations

To make the mechanism credible:

| Ablation | Purpose |
| --- | --- |
| EAGLE-only | Primary baseline |
| existing `sam_sequence_graft` | Current best simple SAM + EAGLE baseline |
| always-on SAM graft | Shows whether conditional activation matters |
| random boundary graft | Rules out budget-only effects |
| depth-only graft | Tests whether concrete node/path prediction matters |
| margin-only node graft | First interpretable predictor |
| risk-score node graft | Stronger non-learned predictor |
| oracle boundary graft | Upper bound |

Expected qualitative ordering:

```text
oracle boundary > risk-score node > margin-only node > depth-only > random
```

If this ordering does not hold, the boundary-repair mechanism is probably not
the right explanation.

## Main Risks

1. EAGLE margin may not be calibrated against target-model rejection.
2. "High-confidence wrong" may exist, but it must be measured separately from
   ordinary low-margin uncertainty.
3. SAM lookup and state-transfer overhead may erase gains unless trigger rate is
   low.
4. Benefits may be HumanEval-specific because code has strong local repetition
   and structural templates.
5. Depth-only implementation can accidentally graft SAM onto the wrong branch.

## Implementation Guardrails for Later

When this moves to code:

- Do not implement a depth-only predictor as the final method.
- Do not use probability thresholds above 1 for `exp(logprob)` scores.
- Keep `tree_method`, `tree_fusion`, and `fusion_mode` semantics separate.
- Keep the verifier unchanged.
- Add CLI/config/test coverage together with any new runtime option.
- Record fusion metadata for skipped and triggered steps so analysis can verify
  trigger rate, predicted boundary, SAM nodes added, and accepted SAM nodes.

## Related Notes

- `2026-06-10-new-directions-codex-gemini.md`: original multi-direction
  brainstorming.
- `2026-06-10-rejection-boundary-novelty-analysis.md`: preliminary novelty and
  related-work positioning.

## Next Step

Design the offline calibration experiment. The first deliverable should be an
analysis script and trace schema, not an online decoding change.
