# Phase A2 results: horizons, extension, and the composite operating point

Four batches on HumanEval 0-164 (Llama-3.1-8B + EAGLE3, greedy, theta=0.86,
sequence path always at the author's n_predicts=40).

## Batch history

1. **horizon** (author 40-horizon for graft+extend): MAT up to +8.75%
   (g40_k2_ext) but throughput negative (0.917-0.985). Profile verdict:
   verify grows ~15% at 141 vs 61 tokens — capacity is compute-priced past
   ~80-100 tokens. Node value ordering: extension > repair > deep EAGLE
   branches (ext +7.38% vs g40 +5.59% for the same ~40 nodes).
2. **realloc** (single n_predicts knob + eagle3_total_token shrink): best
   e60_np24 +4.89%/1.0142. Exposed the knob conflation — np16 cut the
   SEQUENCE draft from 40 too and lost strong-match accepts (+1.83% only).
   Shrinking EAGLE (e45_none -1.63% MAT) not worth it inside the free zone.
3. **composite** (per-mechanism horizons, sequence fixed at 40):

| arm | max tree | MAT | MAT+% | tok/s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 61 | 7.4849 | +0.00 | 215.46 | 1.0000 |
| r8_e16 | 77 | 7.9075 | +5.65 | 219.80 | 1.0201 |
| r8_e24 | 85 | 7.9128 | +5.72 | 219.65 | 1.0194 |
| **r8k2_e16** | **77** | **7.9550** | **+6.28** | **219.75** | **1.0199** |
| r8k2_e24 | 85 | 7.9628 | +6.38 | 218.69 | 1.0150 |
| r12_e24 | 85 | 7.9471 | +6.17 | 215.75 | 1.0013 |

## Operating point (user cost rule: extra verify must be paid by accepts)

**theta=0.86, graft_horizon=8, K=2, extend_horizon=16, sequence=40 →
MAT +6.28%, throughput +2.0%, max tree 77 nodes** — below the
sam_sequence_graft precedent (61 + n_predicts ≈ 101 on match steps).
e24/r12 variants rejected: their MAT deltas do not pay their verify cost.

```bash
--fusion_mode drafter_mars --drafter_mars_repair graft --drafter_mars_theta 0.86 \
--drafter_mars_max_grafts 2 --drafter_mars_graft_horizon 8 \
--drafter_mars_extend --drafter_mars_extend_horizon 16
```

## Method summary (paper narrative)

Uncertainty-routed SAM capacity: per step, the MARS ratio on the greedy top
path routes SAM's n_predicts-class capacity either as *repair* (grafted at
the up-to-K earliest uncertain parents, short horizon) or as *extension*
(grafted at the greedy leaf beyond the EAGLE depth frontier, medium horizon);
strong matches keep the author's full sequence draft. All grafts use the
author's walk-reuse graft semantics; verify-cost data fixes the horizons
(learned per-site prediction of them = Phase B).

## Conventions

- Baseline runs are NOT repeated per batch anymore (user decision); reuse
  `comp_baseline.jsonl`. Revisit only if a speedup conclusion sits within
  the ±1.5% cross-boot drift band.
- Failed detours recorded for the paper: budget schedules (Phase A),
  per-element tensor buffer build (reverted, cf7b8d7..1a60205), single-knob
  horizon (realloc batch).

## Next

- Cross-domain batches (MT-Bench / GSM8K / domain-corpus QA) — decides the
  paper's claim scope; adaptive-theta (ACI) arm rides along.
- Phase B learned head: predict (trigger, route, horizon) per site.
