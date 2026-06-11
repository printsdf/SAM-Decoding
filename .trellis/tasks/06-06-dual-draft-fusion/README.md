# Dual Draft Fusion Task Dashboard

Last updated: 2026-06-11

This is the active Trellis workspace for `dual-draft-fusion`. Read this file
first, then follow only the current links below.

## Current State

Depth-decoupled fusion is closed, and HumanEval oracle evidence approved
Rejection-Boundary SAM Repair for Phase B implementation.

| Dataset | Decision | Evidence |
| --- | --- | --- |
| MedQA q0-80 | Reject | Best depth-decoupled oracle gap `+0.00%` |
| MT-Bench q0-80 | Reject | Best depth-decoupled oracle gap `+0.61%`, below the 3% gate |
| HumanEval q0-164 | Proceed | Rejection-boundary oracle gap `+9.65%`, above the 7% gate |

The active question is no longer "should SAM be fused broadly with EAGLE3?"
It is:

```text
Can predicted-boundary SAM grafting recover enough of the HumanEval oracle
ceiling to beat EAGLE3-only throughput?
```

## Read Next

| Order | File | Why |
| ---: | --- | --- |
| 1 | `prd.md` | Current objective, gates, and non-goals |
| 2 | `docs/experiments/README.md` | Experiment table and status |
| 3 | `docs/experiments/results/2026-06-10-humaneval-oracle-results.md` | Phase B gate evidence |
| 4 | `docs/experiments/specs/2026-06-11-phase2-boundary-graft.md` | Current implementation design |
| 5 | `docs/research/2026-06-10-rejection-boundary-novelty-analysis.md` | Prior-work positioning |

## Active Work

Phase B: implement and validate predicted-boundary SAM grafting on HumanEval.

| Workstream | Decision Gate | Output |
| --- | --- | --- |
| Boundary-graft implementation | MAT `> 7.96` on HumanEval full | Proceed to paper ablations |
| Threshold sweep | Best MAT without TPS collapse | Select `boundary_graft_threshold` |
| Cross-check | MT-Bench q0-80 smaller but non-negative gain | Confirm domain specificity |

Do not use trace-on/profile outputs as final throughput evidence. Profile traces
are diagnostic inputs; speedup claims require trace-off inference outputs.

## Where Things Go

| Path | Use |
| --- | --- |
| `README.md` | This dashboard only |
| `prd.md` | Current research objective and decision rules |
| `docs/experiments/` | Specs, plans, and result notes |
| `docs/research/` | Current research notes that inform active decisions |
| `docs/reference/` | Migrated usage guides and structural notes tied to this task |
| `docs/archive/` | Historical implementation specs, old research, debug notes, old PRDs |
| `scripts/` | Task-local helper scripts |

The task root should not contain ad hoc result files, `research/`, `notes/`, or
old phase docs. Put them under `docs/`.

## Important Caveats

- Use same-trace oracle comparisons for decision gates.
- Use trace-off inference outputs for speedup claims; profile traces are
  diagnostics.
- Do not compare throughput across machines unless provenance is documented.
- Do not build Static SAM memory from evaluation answers or test labels.

## Suggested Opening Prompt

```text
Read .trellis/tasks/06-06-dual-draft-fusion/README.md and prd.md.
Continue Phase B predicted-boundary SAM grafting from docs/experiments/specs/2026-06-11-phase2-boundary-graft.md.
```
