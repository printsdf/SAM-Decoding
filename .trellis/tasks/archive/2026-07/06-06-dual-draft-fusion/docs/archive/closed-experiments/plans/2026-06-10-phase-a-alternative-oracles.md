# Phase A Alternative Oracle Plan

**Goal:** Test two narrow SAM-use hypotheses before any new runtime fusion
implementation.

**Baseline:** `eagle_only` on the same MT-Bench q0-80 fusion profile trace.

**Primary Metrics:** Rejection-boundary conditional oracle gap and
high-precision SAM slice oracle gap.

**Budget:** Offline analysis on one existing MT-Bench profile trace. No model
run is required unless trace metadata is insufficient.

---

## Inputs

```text
evaluation/data/mt_bench/profile_fusion_overhead/naive_fusion_mt_bench_q0_80.fusion_profile.json
```

## Step 1: Confirm Trace Shape

Check that the trace has candidate paths, sources, depths, and acceptance paths.
For the high-precision probe, also check whether SAM match metadata is present.

Stop if the trace is missing required fields and record the missing metadata in
the result note.

## Step 2: Rejection-Boundary Probe

Create or run an offline analyzer that computes:

- first EAGLE rejection depth per step;
- whether recorded SAM candidates can match the accepted suffix from that
  boundary;
- conditional MAT/gap on eligible steps;
- coverage: percentage of steps where the probe is eligible.

Suggested output:

```text
evaluation/data/mt_bench/profile_fusion_overhead/rejection_boundary_oracle.json
```

Decision gate: conditional oracle gap `> 7%`.

## Step 3: High-Precision SAM Probe

Create or run an offline analyzer that filters SAM candidates by match quality,
starting with `match_length >= 10` if the trace has match metadata.

Report:

- high-match slice coverage;
- oracle MAT/gap inside the slice;
- overall expected gain after accounting for coverage;
- leakage risk for any memory/source assumption.

Suggested output:

```text
evaluation/data/mt_bench/profile_fusion_overhead/high_precision_sam_oracle.json
```

Decision gate: slice gap `> 8%` with useful coverage.

## Step 4: Record Result

Write one result note under:

```text
.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/
```

The note must choose one decision:

- implement rejection-boundary repair;
- implement high-precision SAM memory/gating;
- rerun with richer traces;
- reject both probes.

## Stop Conditions

- Do not implement runtime code before a probe passes its gate.
- Do not use evaluation answers or hidden labels to build SAM memory.
- Do not report speedup from profile traces.
