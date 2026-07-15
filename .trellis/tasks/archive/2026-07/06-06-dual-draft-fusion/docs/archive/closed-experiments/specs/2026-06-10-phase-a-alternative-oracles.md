# Phase A Alternative Oracle Spec

## Goal

Quickly test whether SAM has value in narrower situations after
depth-decoupled fusion failed on MedQA and MT-Bench.

## Baseline To Beat

Primary baseline: `eagle_only` on the same MT-Bench q0-80 fusion profile trace.

Context baselines:

- `sam_sequence_graft`
- closed depth-decoupled oracle result
- `perfect` oracle over recorded candidates

## Probe 1: Rejection-Boundary SAM Repair

Hypothesis: If SAM is evaluated at EAGLE's first rejection boundary, then SAM
may recover accepted suffixes that tail-after-success strategies cannot reach.

Metric: conditional oracle gap on steps with an EAGLE rejection boundary.

Selection rule:

```text
conditional gap > 7% -> write implementation plan
conditional gap <= 7% -> reject or rerun only if trace fields are insufficient
```

Expected analysis artifact:

```text
evaluation/data/mt_bench/profile_fusion_overhead/rejection_boundary_oracle.json
```

## Probe 2: High-Precision SAM Slice

Hypothesis: If SAM candidates are filtered to long, source-safe matches, then
the high-precision slice may justify stricter gating or a new memory build.

Metric: oracle gap on high-match SAM slices and coverage of those slices.

Selection rule:

```text
slice gap > 8% with useful coverage -> write memory/gating implementation plan
slice gap <= 8% or negligible coverage -> reject for MT-Bench
```

Expected analysis artifact:

```text
evaluation/data/mt_bench/profile_fusion_overhead/high_precision_sam_oracle.json
```

## Dataset And Split

Use the existing MT-Bench q0-80 fusion profile trace:

```text
evaluation/data/mt_bench/profile_fusion_overhead/naive_fusion_mt_bench_q0_80.fusion_profile.json
```

## Required Inputs

The trace must include:

- EAGLE candidate token paths and depths;
- SAM candidate token paths, depths, scores, and match metadata if available;
- verifier acceptance path per decode step.

If match metadata is absent, the high-precision probe must report that the trace
is insufficient instead of fabricating a proxy.

## Failure Modes

- Existing trace lacks enough metadata to locate rejection boundaries.
- SAM candidates are too sparse at rejection boundaries.
- High-match SAM slices have strong local gap but negligible coverage.
- A probe passes only on trace-on diagnostic data and has no plausible
  trace-off runtime path.

## Next Decision

Exactly one of these should happen after the result note:

- implement rejection-boundary repair;
- implement high-precision SAM memory/gating;
- rerun with richer trace metadata;
- reject both and move to structural macro-grafting or another dataset.
