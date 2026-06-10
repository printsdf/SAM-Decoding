# Experiment Index

Use the superpowers-style flow for new research work:

```text
spec -> plan -> run -> result -> keep/rerun/debug/reject
```

Every experiment result should be linked back to the spec or plan that defined
its baseline, metric, dataset split, and decision rule.

## Active Experiment

| Slug | Spec | Plan | Result |
| --- | --- | --- | --- |
| `2026-06-10-depth-decoupled-oracle` | `specs/2026-06-10-depth-decoupled-oracle.md` | `plans/2026-06-10-depth-decoupled-oracle.md` | Pending MT-Bench result |

## Recorded Results

| Result | Decision |
| --- | --- |
| `results/2026-06-10-medqa-depth-decoupled-oracle.md` | Reject depth-decoupled fusion for MedQA |
| `results/2026-06-09-mtbench-naive-logprob-profile.md` | Do not claim Phase 3.1 improvement; no-tail oracle gate still pending |

## Rules

- Keep benchmark comparisons same-machine and same-trace unless explicitly
  documented otherwise.
- Keep speedup claims separate from profiling or trace-on diagnostics.
- Record negative results with enough detail to avoid repeating them.
- Put raw, stale, or one-off debugging instructions in `../archive/`.
