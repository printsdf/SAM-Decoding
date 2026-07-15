# Experiments

## Active

| Experiment | Status | Spec | Plan | Result |
| --- | --- | --- | --- | --- |
| Drafter-MARS SAM gate | **Implemented + modularized; smoke numbers pending re-run** | [spec](specs/2026-06-11-drafter-mars-sam-gate.md) | [plan](plans/2026-06-11-drafter-mars-sam-gate.md) | [session report 2026-07-13](results/2026-07-13-dual-draft-fusion-report.md) |

## Baselines

| Experiment | Result | Notes |
| --- | --- | --- |
| HumanEval oracle | [result](results/2026-06-10-humaneval-oracle-results.md) | Original +9.65% not reproducible; canonical baseline +2.72% |
| Canonical question file | [result](results/2026-06-16-humaneval-canonical-question-file.md) | 164 questions, SHA256 `fc49f930...` |

## Session notes

| Note | Path |
| --- | --- |
| Full experiment result report (2026-07-13) | [report](results/2026-07-13-dual-draft-fusion-report.md) |
| Cleanup candidate list (identify only) | [cleanup](results/2026-07-13-cleanup-candidates.md) |

## Archived

All closed experiments moved to `archive/closed-experiments/`.
Prior unconditioned low-margin failure:
[2026-06-11-boundary-predictor-calibration](../archive/closed-experiments/results/2026-06-11-boundary-predictor-calibration.md).

## Rules

- Every result must include baseline, metric, confounders, decision, next step.
- Same-trace oracle comparisons for direction gates.
- Trace-off inference for throughput claims, not profile traces.
