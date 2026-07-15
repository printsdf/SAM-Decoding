# Depth-Decoupled Oracle Experiment Spec

## Goal

Measure whether depth-decoupled SAM tail extension has enough oracle ceiling to
justify implementing a runtime fusion mode.

## Hypothesis

If SAM contains useful long-tail continuations after a correct EAGLE prefix,
then the depth-decoupled oracle should improve MAT over EAGLE-only by more than
3% on MT-Bench, because EAGLE covers shallow uncertain tokens while SAM recovers
reusable deeper suffixes.

## Baseline To Beat

Primary baseline: `eagle_only` on the same fusion profile trace.

Context baselines:

- `sam_sequence_graft`
- `perfect` oracle
- `budgeted` oracle
- `source_balanced` oracle

## Metric And Selection Rule

Primary metric: oracle MAT gap versus `eagle_only`.

Selection rule:

```text
gap > 3%  -> consider implementing depth-decoupled runtime fusion
gap < 3%  -> reject this direction for that benchmark
```

MedQA already failed this rule with `+0.00%`; MT-Bench is the active gate.

## Dataset And Split

| Dataset | Split | Status |
| --- | --- | --- |
| MedQA | q0-80 fusion profile trace | Completed, negative |
| MT-Bench | q0-80 fusion profile trace | Pending |

The profile trace must contain EAGLE and SAM candidate paths plus verifier
acceptance paths, compatible with `evaluation/oracle_fusion_analysis.py`.

## Expected Code Or Config Changes

No runtime fusion implementation should be added before the MT-Bench oracle gate
passes.

Required analysis code already exists:

- `evaluation/oracle_depth_decoupled.py`
- `evaluation/oracle_fusion_analysis.py`
- `scripts/profile_fusion_mt_bench.sh`

## Sanity Checks

- Verify the trace was produced with `--profile-fusion`.
- Verify the analyzer reports `eagle_only` and `depth_decoupled_d<N>` entries.
- Verify MAT includes the root/start token, matching the shared oracle contract.
- Verify whether EAGLE3 tail sidecar was enabled; do not mix tail-enabled and
  no-tail decisions.

## Failure Modes

- SAM candidate paths rarely match accepted EAGLE prefixes.
- The perfect oracle ceiling is below the decision gate.
- The profile trace is tail-enabled when the intended gate is no-tail.
- Remote runtime provenance is unclear, making throughput comparisons invalid.

## Next Decision

If MT-Bench oracle gap passes the gate, write an implementation plan for
`tree_fusion="eagle_leaf_sam_extend"`. If it fails, pivot to
rejection-boundary recovery, delayed leaf sidecar, or a negative-result story.
