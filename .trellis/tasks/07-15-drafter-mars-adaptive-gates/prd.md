# PRD: Adaptive and learned gates for drafter-MARS

Status: planning approved (user 2026-07-15)
Task: `drafter-mars-adaptive-gates`
Baseline operating point (archived task 06-06): online drafter_mars, theta=0.86,
graft budget=8 → MAT +4.01%, tok/s +2.61% vs SAM[EAGLE3] on HumanEval 0-164.

## Problem

The fixed-theta, fixed-budget, single-graft gate leaves three knobs static:

1. theta is domain-sensitive (trigger rate 75.6% on HumanEval at 0.90; unknown
   elsewhere) — a fixed value will not transfer across benches.
2. Graft budget ignores WHERE the trigger fired (deep trigger → few tree levels
   left to repair → budget wasted in verify cost).
3. Only the earliest triggering parent is repaired; other uncertain top-path
   parents are ignored.

And the ratio gate itself is a hand-crafted zero-shot signal; a tiny learned
head on drafter states may separate reject/accept boundaries better.

## Goals

- **Phase A (adaptive trio)**: online theta self-calibration to a target
  trigger rate; budget as a function of trigger depth/ratio; multi-point graft.
  Each independently switchable; all defaults reproduce the current operating
  point bit-identically.
- **Phase B (learned gate)**: capture per-parent features (+ optional drafter
  hidden states) with verifier-derived labels, train a tiny gate head offline,
  serve it online behind the same gate interface; MARS ratio remains the
  zero-shot fallback/baseline.

## Hard constraints

1. **Deletability**: all new logic lives in dedicated modules
   (`samd/fusion/drafter_mars_adaptive.py`, `samd/fusion/gate_head.py`,
   `evaluation/gate_head/`); existing files gain only config fields, CLI
   flags, and a few guarded hook lines. Removing the modules + reverting hooks
   restores the current behavior exactly.
2. **Default equivalence**: with all new switches off, outputs are
   bit-identical to commit `fbd338c` behavior (MAT 7.7849 on the t086 rerun is
   the acceptance check).
3. torch-free unit tests for all pure logic; server GPU only for eval arms.
4. No samd changes outside the drafter_mars branch + config + fusion modules.

## Acceptance criteria

- [ ] A: unit tests for controller direction/clipping, budget modes,
      multi-trigger selection; server batch shows each switch's MAT/tok-s vs
      the fixed operating point; adaptive theta holds trigger rate within
      ±5pp of target on HumanEval without MAT collapse.
- [ ] B: capture run produces a labeled dataset; trained head beats the ratio
      gate on held-out AUC; online arm runs end-to-end and reports MAT/tok-s
      vs the ratio gate.
- [ ] Default-off equivalence rerun passes (MAT 7.7849).
- [ ] Results recorded in task docs; batch scripts with Feishu notify +
      auto-shutdown.

## Out of scope

- Cross-domain benches (separate task after this one).
- vLLM / batch>1 integration.
- Verifier-side relaxed acceptance (MARS proper).
