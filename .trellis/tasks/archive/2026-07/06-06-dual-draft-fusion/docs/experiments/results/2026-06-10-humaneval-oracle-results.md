# HumanEval Oracle Results

## Original Oracle (2026-06-10, Provenance Lost)

**Decision**: ⚠️ NOT REPRODUCIBLE — original question file and config lost.

| Method | MAT | Nodes | MAT/node | Oracle gap |
|--------|-----|-------|----------|-----------|
| eagle3_only | 3.6689 | 59.00 | 0.06219 | +0.00% |
| sam_sequence_graft | 3.6812 | 59.15 | 0.06224 | +0.33% |
| perfect | 4.0231 | 59.15 | 0.06802 | +9.65% |
| depth_decoupled | 3.7535 | 63.48 | 0.05913 | +2.30% |

## Canonical HumanEval Validation (2026-06-18)

Source: standard HumanEval 164 problems (SHA256: `fc49f930...`)

| Method | MAT | Nodes | MAT/node | Oracle gap |
|--------|-----|-------|----------|-----------|
| eagle3_only | 6.8180 | 59.00 | 0.11556 | +0.00% |
| sam_sequence_graft | 6.8288 | 59.26 | 0.11524 | +0.16% |
| perfect | 7.0036 | 59.26 | 0.11819 | +2.72% |
| depth_decoupled | 6.8968 | 66.85 | 0.10317 | +1.16% |

Rejection-boundary oracle:

| Metric | Original | Canonical |
|--------|----------|-----------|
| Total steps | 6932 | 8143 |
| EAGLE rejection steps | 432 (6.2%) | 390 (4.8%) |
| SAM rescue rate | 100% | 100% |
| Oracle gap | +9.65% | **+2.72%** |

## Provenance Gap

The original run cannot be reproduced. Three different question files existed:

| Trace | Rows | SHA256 (prefix) | EAGLE MAT |
|-------|------|-----------------|-----------|
| Original (lost) | ≥164 | unknown | 3.67 |
| Old q0-20 smoke | 20 | `d49a763b` | 6.79 |
| Canonical (current) | 164 | `fc49f930` | 6.82 |

The MAT difference (3.67 vs 6.82) cannot be explained by `max_new_tokens`
alone. Most likely cause: different question content or prompt formatting in the
original lost file.

## Current Baseline

Use canonical HumanEval as the ground truth:
- EAGLE MAT: **6.82**
- Oracle ceiling: **+2.72%**
- This is the baseline Drafter-MARS must beat.
