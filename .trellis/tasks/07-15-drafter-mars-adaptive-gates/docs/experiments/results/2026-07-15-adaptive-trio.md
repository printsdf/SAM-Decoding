# Phase A results: adaptive trio on HumanEval (2026-07-15)

Batch: `scripts/run_drafter_mars_adaptive_batch.sh`, HumanEval 0-164, greedy,
Llama-3.1-8B-Instruct + EAGLE3, theta=0.86, base budget 8, repair=graft.
Reference arm `adaptA_ref` = default-off; equivalence vs commit `546f870`
lineage confirmed (MAT = 7.7849 exactly). MAT+%/speedup below are relative to
`adaptA_ref`; vs the SAM[EAGLE3] baseline (7.4849) add +4.01pp of context.

| arm | MAT | vs ref | tok/s | speedup |
| --- | ---: | ---: | ---: | ---: |
| ref (fixed theta, b8, K=1) | 7.7849 | +0.00 | 220.18 | 1.0000 |
| adaptive theta target 0.75 | 7.7577 | -0.35 | 219.64 | 0.9975 |
| adaptive theta target 0.60 | 7.7161 | -0.88 | 219.04 | 0.9948 |
| budget mode depth | 7.6297 | -1.99 | 216.60 | 0.9837 |
| budget mode ratio | 7.6043 | -2.32 | 217.14 | 0.9862 |
| **K=2 multi-graft** | **7.8372** | **+0.67** | **220.19** | **1.0000** |
| K=3 multi-graft | 7.8403 | +0.71 | 219.97 | 0.9991 |
| trio (adaptive+depth+K2) | 7.6172 | -2.15 | 216.56 | 0.9835 |

## Findings

1. **K=2 is a free win**: MAT 7.8372 (+4.71% vs SAM[EAGLE3]) at identical
   throughput. K=3 adds only +0.04pp. **New operating point: theta=0.86,
   budget=8, K=2.**
2. **Hand-crafted budget schedules are falsified**: both depth (-1.99%) and
   ratio (-2.32%) modes lose MAT. Root cause: grafts extend BEYOND the EAGLE
   tree frontier (graft at depth d reaches d+budget), so "deeper trigger →
   less room" is wrong as a premise; earlier b12 (+0.27pp over b8) also says
   budget wants to grow, not shrink. This is the empirical case for a
   LEARNED budget (Phase B) and a clean negative ablation for the paper.
3. **Adaptive theta is in-domain neutral-negative** (-0.35% at target 0.75),
   as expected — theta=0.86 was tuned on this domain. Its ACI framing
   (update rule ≡ Adaptive Conformal Inference, Gibbs & Candès 2021) and its
   test belong to the cross-domain task.
4. tok/s spreads are within noise; MAT differences are deterministic.

## Paper positioning locked (user decision)

Learned multi-task decision head (Phase B) = method; MARS ratio = zero-shot
baseline; ACI-calibrated theta = training-free baseline; fixed theta =
ablation; budget-schedule negatives = motivation for learned budget.

## Closeout

- The optional K=2 + budget-12 and cross-domain adaptive-theta follow-ups were
  not pursued after the project closeout decision on 2026-07-17.
