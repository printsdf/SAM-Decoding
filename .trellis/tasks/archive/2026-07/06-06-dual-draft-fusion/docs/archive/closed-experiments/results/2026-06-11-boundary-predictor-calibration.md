# Boundary Predictor Calibration Result Summary

## Claim

- Tested whether node-level EAGLE3 low-margin signals can identify the
  rejection boundary where SAM repair has value.
- The q0-20 HumanEval smoke answers a narrower question: an unconditioned
  "any EAGLE node has low margin" trigger has oracle utility, but it is not a
  usable sparse predictor.

## Evidence

- Baseline run: HumanEval smoke q0-20 enriched fusion profile, EAGLE3 +
  recorded SAM candidates, `node_budget=60`.
- Selection rule: thresholds selected on q0-10 decode steps, evaluated on
  q10-20 decode steps.
- Analyzer config:

```text
max_thresholds: 128
node_budget: 60
train split: q0-10, 863 decode steps
heldout split: q10-20, 896 decode steps
```

- Selected low-margin threshold:

```text
low_margin_threshold: 3.9066619873046875
selected_by: unconstrained_diagnostic
selection_constraints_met: false
```

- Held-out metrics:

```text
trigger_rate: 99.89%
boundary_precision: 25.47%
boundary_recall: 79.44%
eagle_mat: 3.9833
perfect_mat: 4.3449
predicted_boundary_oracle_mat: 4.1406
pass: false
```

- Derived held-out utility:

```text
predicted vs EAGLE: +3.95%
perfect vs EAGLE: +9.08%
recovered oracle ceiling: ~43.5%
```

The same-split oracle accounting is coherent:

```text
3.9833 eagle_mat < 4.1406 predicted_boundary_oracle_mat < 4.3449 perfect_mat
```

## Confounders

- This is a 20-question smoke run, not the planned q0-164 full HumanEval
  calibration.
- The predictor examined low margins anywhere in the EAGLE tree. Large trees
  almost always contain at least one low-margin irrelevant branch, so this
  trigger can degenerate into always-on SAM repair.
- The MAT numbers are offline oracle metrics from trace/profile output. They do
  not include online SAM generation cost or throughput impact.
- The q10-20 held-out MAT values must not be compared directly to q0-20 or
  q0-164 oracle summaries.

## Decision

- Debug and redesign predictor features before any online boundary-graft
  implementation.
- Do not run full HumanEval q0-164 for the current unconditioned low-margin
  trigger as the primary method.
- Keep the trace enrichment and analyzer code because they exposed the failure
  mode and provide the right offline harness for feature search.

## Code Retention

- Keep code changes.
- Relevant commits:

```text
9508a90 Add boundary predictor calibration analyzer
01c1f7e Clarify boundary predictor MAT reporting
```

- Revert condition: discard only if a simpler analyzer with the same enriched
  trace fields replaces this one and reproduces the same schema checks, budget
  accounting, and split reporting.

## Avoid Repetition

- Do not repeat the unconditioned "any node has margin below threshold" rule as
  the main predictor.
- Do not interpret `unconstrained_diagnostic` as a pass, even if MAT improves.
- Do not compare predicted MAT from q10-20 against perfect MAT from q0-20 or
  q0-164.

## Next Step

- Search for path-conditioned features that combine uncertainty with likelihood
  that the verifier will actually reach the node.
- Start with non-learned ablations:
  - top-path-only margins;
  - `rank_among_siblings == 1` candidates only;
  - cumulative path logprob or parent probability filters;
  - risk scores combining low margin, high path probability, and depth prior.
- Only run the full q0-164 calibration after a smoke ablation finds a trigger
  rate near the <= 15% target without losing most oracle utility.
