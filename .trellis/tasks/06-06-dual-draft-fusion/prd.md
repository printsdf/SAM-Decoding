# Depth-Decoupled Dual Draft Fusion PRD

Status: in progress  
Last updated: 2026-06-10  
Task: `dual-draft-fusion`

## Problem

SAM and EAGLE3 can both propose speculative draft tokens, but naive candidate
fusion has not improved throughput or mean accepted tokens. The current research
question is whether a more structured collaboration can expose useful SAM tails
without damaging EAGLE's strong shallow predictions.

## Current Direction

The active idea is depth-decoupled fusion:

- EAGLE owns shallow draft depths up to `D_split`.
- SAM contributes only deeper tail candidates after the accepted EAGLE prefix.
- Offline oracle analysis decides whether this has enough ceiling before any
  new runtime fusion mode is implemented.

This replaced the earlier Phase 3 payoff-aware plan as the current decision
path. The previous long PRDs are archived under `docs/archive/`.

## Hypothesis

If MT-Bench contains reusable suffix patterns that SAM can recover after a good
EAGLE prefix, then a depth-decoupled oracle should improve MAT over EAGLE-only
by more than 3% on the same fusion profile trace, because EAGLE handles local
syntax while SAM contributes longer low-entropy continuations.

## Baselines

Use same-trace comparisons only:

| Baseline | Purpose |
| --- | --- |
| `eagle_only` | Primary baseline for oracle MAT gap |
| `sam_sequence_graft` | Existing SAM insertion strategy for context |
| `perfect` oracle | Ceiling for whether any recorded SAM/EAGLE fusion has value |
| `naive_logprob` | Negative/neutral implemented baseline |

## Primary Metric

Primary metric: oracle MAT gap versus `eagle_only` on the same fusion profile
trace.

Selection rule:

```text
gap > 3%  -> implement or prototype depth-decoupled fusion
gap < 3%  -> reject this direction for the benchmark and pivot
```

For any implemented runtime method, throughput must be measured from trace-off
inference outputs. Fusion-profile or trace-on outputs are diagnostic only.

## Dataset Scope

| Dataset | Status | Decision |
| --- | --- | --- |
| MedQA q0-80 | Completed on 2026-06-10 | Rejected, best depth-decoupled gap `+0.00%` |
| MT-Bench q0-80 | Pending | Active decision gate |

## Completed Work

- Implemented naive fusion baseline and confirmed it is not a candidate default.
- Added EAGLE3 logprob extraction and reran MT-Bench comparisons.
- Added fusion profiling and oracle trace tooling.
- Added `evaluation/oracle_depth_decoupled.py` and unit coverage.
- Ran MedQA depth-decoupled oracle; result rejected the direction for MedQA.

## Active Work

1. Run or locate MT-Bench fusion profile trace for `naive_fusion_mt_bench_q0_80`.
2. Run `evaluation/oracle_depth_decoupled.py` on that trace.
3. Record the result under `docs/experiments/results/`.
4. Decide whether to implement depth-decoupled fusion or pivot.

## Non-Goals

- Do not implement `tree_fusion="eagle_leaf_sam_extend"` before the MT-Bench
  oracle gate passes.
- Do not treat MedQA as a positive result for this direction.
- Do not compare speedup across machines or trace settings.
- Do not rebuild Static SAM artifacts from evaluation benchmark data.

## Risks

- Recorded-candidate oracle does not regenerate SAM from every EAGLE leaf, so
  it is a conservative diagnostic, not proof of runtime behavior.
- Tail-enabled and no-tail EAGLE3 runs are different conditions.
- Repository dirtiness during earlier runs means some historical artifacts are
  not self-contained.

## Next Step

Follow `docs/experiments/plans/2026-06-10-depth-decoupled-oracle.md` for the
MT-Bench oracle gate.
