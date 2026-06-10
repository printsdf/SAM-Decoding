# Dual Draft Fusion Task Index

Last updated: 2026-06-10

This is the active Trellis workspace for `dual-draft-fusion`. Use this file as
the entry point. Detailed experiment design, execution plans, and result notes
live under `docs/experiments/` so future sessions can follow the superpowers
research workflow without reading every historical scratch note.

## Read First

| File | Purpose |
| --- | --- |
| `prd.md` | Current research objective, decision gates, and open work |
| `docs/experiments/README.md` | Experiment index and documentation rules |
| `docs/experiments/specs/2026-06-10-depth-decoupled-oracle.md` | Active experiment design card |
| `docs/experiments/plans/2026-06-10-depth-decoupled-oracle.md` | Remote execution and artifact plan |
| `docs/experiments/results/2026-06-10-medqa-depth-decoupled-oracle.md` | MedQA oracle result and rejection decision |
| `docs/experiments/results/2026-06-09-mtbench-naive-logprob-profile.md` | Phase 3.1 MT-Bench logprob/profiling result |
| `.trellis/spec/backend/evaluation-protocols.md` | Benchmark and trace protocol |
| `.trellis/spec/backend/tree-fusion.md` | Fusion-mode implementation contract |

## Current Bottom Line

`naive_fusion` remains a negative or neutral baseline. Replacing the EAGLE depth
proxy with real logprobs removed the misleading catastrophic interpretation, but
the current-machine MT-Bench subset still trailed pure EAGLE3:

```text
pure_eagle3_current_q0_11:      54.109 TPS, 6.189 MAT
naive_logprob_current_q0_11:    52.047 TPS, 6.075 MAT
relative throughput:             0.962x
```

The 2026-06-10 MedQA depth-decoupled oracle rejected that direction for MedQA:
the best split matched EAGLE-only at `+0.00%` oracle gap, while the perfect
oracle ceiling was only `+1.45%`.

The next valid decision is the MT-Bench depth-decoupled oracle gate:

```text
oracle gap > 3%  -> consider implementing depth-decoupled fusion
oracle gap < 3%  -> pivot to rejection-boundary recovery or another direction
```

## Directory Map

| Path | Status | Notes |
| --- | --- | --- |
| `prd.md` | Current | Short PRD and handoff summary |
| `task.json` | Current | Trellis metadata |
| `implement.jsonl`, `check.jsonl` | Current | Curated context for future non-inline agents |
| `docs/experiments/specs/` | Current | Hypotheses, baselines, metrics, and decision gates |
| `docs/experiments/plans/` | Current | Exact remote commands, artifacts, sanity checks, stop conditions |
| `docs/experiments/results/` | Current | Evidence, confounders, decisions, and next steps |
| `docs/implementation/` | Historical/current | Implementation designs and summaries by phase |
| `docs/research/` | Historical/current | Literature, diagnosis, and comparative analysis |
| `docs/archive/` | Historical | Stale PRDs, debug checklists, validation commands, and session summaries |
| `scripts/` | Historical/current | Task-local remote/evaluation helpers |

## Documentation Rules

- Keep current status in `README.md` and `prd.md`; do not create another
  top-level status file.
- Put each new experiment in `docs/experiments/` with a date-prefixed slug.
- A result note must state the baseline, metric, selection rule, confounders,
  decision, and next step.
- Historical debug instructions belong under `docs/archive/`, not the task root.
- Preserve remote artifact paths and machine/runtime caveats in result notes.

## Reproducibility Caveats

- The repository was dirty during Phase 3.1 validation. Do not assume commit
  `09c97b5` alone reproduces the run.
- Full-run MT-Bench files with 140+ TPS should not be compared against current
  50+ TPS runs unless machine/runtime provenance is proven.
- Use trace-off outputs for speedup comparisons. Trace-on or fusion-profile
  outputs are diagnostic and should not become final throughput claims.
- `scripts/profile_fusion_mt_bench.sh` passes EAGLE3 tail flags if the
  environment defines `EAGLE3_TAIL_PATH` or `EAGLE3_TAIL_TYPE`; unset them for a
  no-tail oracle gate.

## Suggested Opening Prompt

```text
Read .trellis/tasks/06-06-dual-draft-fusion/README.md and prd.md.
Continue the MT-Bench depth-decoupled oracle gate using docs/experiments/.
```
