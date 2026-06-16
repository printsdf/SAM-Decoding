# Dual Draft Fusion Task Dashboard

Last updated: 2026-06-16

This is the active Trellis workspace for `dual-draft-fusion`. Read this file
first, then follow only the current links below.

## Current State

Depth-decoupled fusion is closed, and HumanEval oracle evidence approved
Rejection-Boundary SAM Repair as the Phase B direction.

The active implementation gate is **not** online boundary grafting yet. The
latest **Drafter-MARS SAM gate** offline smoke experiment failed its q0-20 gate
and should not be escalated to full calibration yet:

```text
HumanEval q0-20
train: q0-10
heldout: q10-20
predictor-suite: drafter_mars
heldout_trigger_rate: 13.21%
heldout_boundary_precision: 21.43%
heldout_predicted_boundary_oracle_mat: 7.1396
heldout_eagle_mat: 7.0811
predicted MAT gain: +0.83% (< +2% smoke gate)
same-trace perfect gap: +3.38% (suspiciously below prior HumanEval +9.65%)
same-trace rejection-boundary gap: +3.91% overall q0-20
```

This follows the failed unconditioned low-margin boundary-predictor smoke, which
reached useful oracle MAT but triggered on `99.89%` of held-out decode steps.
That result is diagnostic only; it should not be repeated as the main method.

The Drafter-MARS run triggered sparsely enough and met the precision gate, but
its held-out MAT gain was too small and its same-trace perfect ceiling was much
lower than prior HumanEval oracle evidence. After the missing auxiliary oracle
scripts were synced into the experiment worktree, the q0-20 rejection-boundary
oracle generated successfully and matched the perfect ceiling (`+3.91%`
overall). Treat this as a failed smoke for the current q0-20 trace/split, not
as a final rejection of all Drafter-MARS-style predictors. Do not run full
q0-164 calibration or online boundary grafting until the low q0-20 ceiling is
explained or a new smoke trace shows a useful ceiling.

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
| 3 | `docs/experiments/results/2026-06-11-drafter-mars-sam-gate.md` | Why the current Drafter-MARS gate is inconclusive |
| 4 | `docs/experiments/specs/2026-06-11-drafter-mars-sam-gate.md` | Experiment design |
| 5 | `docs/experiments/plans/2026-06-11-drafter-mars-sam-gate.md` | Execution plan and expected sanity checks |
| 6 | `docs/experiments/results/2026-06-11-boundary-predictor-calibration.md` | Why the previous low-margin trigger failed |
| 7 | `docs/experiments/results/2026-06-10-humaneval-oracle-results.md` | Phase B gate evidence |
| 8 | `docs/research/2026-06-10-rejection-boundary-novelty-analysis.md` | Prior-work positioning |
| 9 | `docs/research/2026-06-11-rejection-boundary-sam-repair-proposal.md` | Proposal and margin/threshold semantics |

## Active Work

Phase B: first calibrate predicted rejection-boundary signals on HumanEval, then
implement online grafting only if the offline signal and oracle-utility gates
pass.

| Workstream | Decision Gate | Output |
| --- | --- | --- |
| Drafter-MARS SAM gate | Failed q0-20 smoke: trigger rate `13.21%`, precision `21.43%`, predicted MAT gain `+0.83%`, but perfect/rejection-boundary ceiling only `+3.91%` overall (`+3.38%` held-out) | Debug low trace ceiling before any full q0-164 |
| Boundary predictor calibration | Failed for current unconditioned low-margin trigger: `99.89%` held-out trigger rate | Keep analyzer; do not proceed to online graft from this rule |
| Boundary-graft implementation | Requires a future offline predictor gate to pass before coding | Blocked for Drafter-MARS |
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
Continue Phase B by auditing the Drafter-MARS smoke trace/provenance; do not run
full q0-164 or online grafting until the same-trace oracle ceiling is explained.
```
