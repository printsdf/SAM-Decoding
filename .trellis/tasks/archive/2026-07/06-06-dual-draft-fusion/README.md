# Dual Draft Fusion Task

Last updated: 2026-07-13

## Current State

Drafter-MARS offline path is **implemented and modularized**.
Canonical HumanEval baseline: EAGLE MAT 6.82, oracle ceiling +2.72%.
Smoke held-out numbers are still pending a remote re-run on the
canonical question file.

## Read Order

| # | File | Purpose |
| ---: | --- | --- |
| 1 | `prd.md` | Objective, gates, acceptance criteria |
| 2 | `design.md` | Offline modularization layout (three packages) |
| 3 | `implement.md` | Modularization + cleanup checklist |
| 4 | `docs/experiments/results/2026-07-13-dual-draft-fusion-report.md` | Session result report |
| 5 | `docs/experiments/specs/2026-06-11-drafter-mars-sam-gate.md` | Experiment design |
| 6 | `docs/experiments/plans/2026-06-11-drafter-mars-sam-gate.md` | Remote smoke/full commands |

## Offline package layout

```text
evaluation/oracle/         # types, load, select, analyze, rejection_boundary
evaluation/boundary/       # Prediction, metrics, selection, low_margin, CLI
evaluation/drafter_mars/   # parents, predictors, schema, calibrate
```

CLI:

```bash
python -m evaluation.oracle.cli ...
python -m evaluation.oracle.rejection_boundary ...
python -m evaluation.boundary.cli --predictor-suite drafter_mars ...
```

Legacy top-level modules remain as thin shims.

## Implementation Checklist

- [x] Raw-logit capture + trace enrichment
- [x] Oracle loader parent logit fields
- [x] Drafter-MARS suite + Q2 guard
- [x] Synthetic local tests (25 offline tests pass)
- [x] Offline modularization into three packages
- [x] Delete unrelated evaluation/scripts residuals
- [ ] Remote q0-20 smoke on canonical question file
- [ ] Remote analyzer sweep + smoke decision note

## Important Notes

- Original +9.65% oracle evidence cannot be reproduced. Use canonical baseline.
- samd runtime was intentionally left intact (smoke capture depends on it).
- Smoke provenance previously used question file SHA256 `d49a763b...`;
  re-run should use canonical `fc49f930...`.
