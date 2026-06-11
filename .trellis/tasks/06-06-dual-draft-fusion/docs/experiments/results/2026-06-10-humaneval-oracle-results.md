# HumanEval Oracle Results (2026-06-10)

## Summary

**Decision**: ✅ PASS — all gates cleared, proceed to Phase B implementation.

## Perfect Oracle

| Method | MAT | Nodes | MAT/node | Oracle gap |
|--------|-----|-------|----------|-----------|
| eagle3_only | 3.6689 | 59.00 | 0.06219 | +0.00% |
| sam_sequence_graft | 3.6812 | 59.15 | 0.06224 | +0.33% |
| perfect | 4.0231 | 59.15 | 0.06802 | **+9.65%** |
| depth_decoupled | 3.7535 | 63.48 | 0.05913 | +2.30% |

## Rejection-Boundary Oracle

| Metric | Value |
|--------|-------|
| Total steps | 6932 |
| EAGLE rejection steps | 432 (6.2%) |
| SAM rescue success | 432 |
| SAM rescue rate | **100.0%** |
| Baseline MAT | 3.6689 |
| Oracle MAT | 4.0231 |
| **Oracle gap** | **+9.65%** |

Key insight: rejection-boundary gap = perfect gap. All SAM value is at EAGLE
failure points.

## High-Precision SAM Oracle

| Threshold | SAM Count | Coverage | Oracle MAT | Gap |
|-----------|-----------|----------|-----------|-----|
| baseline | - | - | 2.6689 | +0.00% |
| 5 | 5.06 | 14.6% | 2.9932 | **+12.15%** |
| 8 | 4.62 | 14.6% | 2.9645 | +11.08% |
| 10 | 4.33 | 14.5% | 2.9321 | +9.86% |
| 12 | 4.04 | 14.5% | 2.8925 | +8.38% |
| 15 | 3.60 | 14.5% | 2.8153 | +5.49% |
| 20 | 2.88 | 14.5% | 2.7077 | +1.45% |

Optimal threshold: 5 (gap: +12.15%)

## Cross-Dataset Comparison

| Dataset | Perfect gap | Rejection gap | High-precision gap | Decision |
|---------|------------|--------------|-------------------|----------|
| MedQA | +1.45% | - | - | Reject |
| MT-Bench | +3.01% | +3.01% | +3.42% | Reject |
| **HumanEval** | **+9.65%** | **+9.65%** | **+12.15%** | **✅ Pass** |

## Fusion Overhead

- fusion_overhead_pct: 1.95%
- SAM draft time: 0.03s / 6932 steps (negligible)
- Fusion logic: 5.56s (0.8ms/step)

## Next Step

Proceed to Phase B: implement Rejection-Boundary SAM Repair on HumanEval.
