# Experiment Index

Use the superpowers-style flow:

```text
spec -> plan -> run -> result -> keep/rerun/debug/reject
```

## Active

| Experiment | Spec | Plan | Result |
| --- | --- | --- | --- |
| Drafter-MARS SAM gate | `specs/2026-06-11-drafter-mars-sam-gate.md` | `plans/2026-06-11-drafter-mars-sam-gate.md` | Pending |
| Boundary predictor calibration | `specs/2026-06-11-boundary-predictor-calibration.md` | `plans/2026-06-11-boundary-predictor-calibration.md` | `results/2026-06-11-boundary-predictor-calibration.md` |

## Closed

| Experiment | Result | Decision |
| --- | --- | --- |
| HumanEval rejection-boundary oracle | `results/2026-06-10-humaneval-oracle-results.md` | Proceed, rejection-boundary gap `+9.65%` |
| MT-Bench depth-decoupled oracle | `results/2026-06-10-mtbench-depth-decoupled-oracle.md` | Reject, gap `+0.61%` < 3% |
| MedQA depth-decoupled oracle | `results/2026-06-10-medqa-depth-decoupled-oracle.md` | Reject, gap `+0.00%` |
| MT-Bench naive logprob profile | `results/2026-06-09-mtbench-naive-logprob-profile.md` | Keep as negative/neutral baseline |
| RerankSpec oracle gap pilot | `results/2026-06-02-rerankspec-oracle-gap-pilot.md` | Historical pilot |
| SAM/EAGLE3 tree union pruning | `results/2026-06-01-sam-eagle3-tree-union-pruning.md` | Historical pilot |
| SAM/EAGLE3 tree fusion | `results/2026-06-01-sam-eagle3-tree-fusion.md` | Historical pilot |
| EAGLE-prefix SAM local expansion | `results/2026-06-01-eagle-prefix-sam-local-expansion.md` | Historical pilot |
| CCA projected OOV probe | `results/2026-05-28-cca-projected-oov-probe.md` | Historical probe |
| Rank-truncated OOV probe | `results/2026-05-28-rank-truncated-oov-probe.md` | Historical probe |
| Spectral vocab recovery rerun | `results/2026-05-28-spectral-vocab-recovery-shifted-rerun.md` | Historical probe |

## Historical Specs And Plans

| Experiment | Spec | Plan |
| --- | --- | --- |
| Superseded Phase 2 depth-only boundary graft | `specs/2026-06-11-phase2-boundary-graft.md` | Superseded by boundary predictor calibration |
| Phase A alternative oracles | `specs/2026-06-10-phase-a-alternative-oracles.md` | `plans/2026-06-10-phase-a-alternative-oracles.md` |
| Rejection-boundary SAM repair | `specs/2026-06-10-rejection-boundary-sam-repair.md` | In spec |
| Depth-decoupled oracle | `specs/2026-06-10-depth-decoupled-oracle.md` | `plans/2026-06-10-depth-decoupled-oracle.md` |
| EAGLE-prefix SAM local expansion | `specs/2026-06-01-eagle-prefix-sam-local-expansion.md` | None |
| SAM/EAGLE3 tree union pruning | `specs/2026-06-01-sam-eagle3-tree-union-pruning.md` | None |
| CCA projected OOV probe | `specs/2026-05-28-cca-projected-oov-probe.md` | `plans/2026-05-28-cca-projected-oov-probe.md` |
| Decoding-step OOV probe | `specs/2026-05-28-decoding-step-oov-probe.md` | `plans/2026-05-28-decoding-step-oov-probe.md` |
| Rank-truncated OOV probe | `specs/2026-05-28-rank-truncated-oov-probe.md` | `plans/2026-05-28-rank-truncated-oov-probe.md` |

## Rules

- Current task status belongs in `README.md` and `prd.md`, not in loose result
  files.
- Every new result must include baseline, metric, selection rule, confounders,
  decision, and next step.
- Keep benchmark comparisons same-machine or same-trace unless documented.
- Keep speedup claims separate from trace-on or fusion-profile diagnostics.
