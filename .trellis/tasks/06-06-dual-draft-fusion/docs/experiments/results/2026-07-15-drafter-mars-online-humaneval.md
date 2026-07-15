# Drafter-MARS online HumanEval results (2026-07-15)

Branch `feature/dual-draft-fusion`, commits `011c5c5..d8cb30e` (online
drafter_mars + graft repair + eval tooling). HumanEval q0-164, greedy,
Llama-3.1-8B-Instruct + EAGLE3 drafter, `len_threshold=5, len_bias=5,
max_cache_len=4096`, graft budget `boundary_graft_max_sam_nodes=8`, profile
off unless noted. Baseline = SAM[EAGLE3] (`fusion_mode=none, tree_fusion=none`).
Greedy decoding is deterministic: repeated runs reproduce identical
tokens/steps/MAT; only wall time varies.

## Theta sweep (single runs)

| model | MAT | MAT+% | tok/s | speedup |
| --- | ---: | ---: | ---: | ---: |
| baseline | 7.4849 | +0.00 | 214.85 | 1.0000 |
| drafter_mars graft t=0.84 | 7.7724 | +3.84 | 216.68 | 1.0085 |
| **drafter_mars graft t=0.86** | **7.7849** | **+4.01** | **217.81** | **1.0138** |
| drafter_mars graft t=0.88 | 7.7347 | +3.34 | 215.77 | 1.0043 |
| drafter_mars graft t=0.90 | 7.7128 | +3.05 | 214.04 | 0.9962 |
| drafter_mars graft t=0.94 | 7.6696 | +2.47 | 214.28 | 0.9973 |
| drafter_mars graft t=0.98 | 7.5847 | +1.33 | 213.72 | 0.9948 |

Inverted-U in theta; peak at 0.86. MAT +4.01% exceeds the Stage-1 same-trace
rejection-boundary oracle ceiling (+2.72%) — online grafting alters
trajectories, so exceeding the fixed-trace ceiling is possible.

## Repeats at the operating point (interleaved with baseline)

baseline tok/s: 214.85 / 214.60 / 215.02 → mean 214.82
t=0.86 tok/s: 217.81 / 216.70 / 217.41 / 217.26 → mean 217.30

Ranges are disjoint; **speedup 1.0115 ± 0.003**. MAT identical across reps
(deterministic outputs).

## Always-on control (theta=0.50, trigger → ~100%)

MAT +2.99%, speedup 1.0008 — worse than t=0.86 on both axes. The ratio gate's
when/where selection contributes ~1pp MAT and ~1pp throughput over
indiscriminate grafting.

## Gate behavior (t=0.90 profile run)

trigger rate 75.6% (5774/7637 steps), earliest-parent depth mean 2.15
(min 0, max 7), SAM nodes grafted mean 6.96/8, skip only 7
(`empty_sam_continuation`). The gate is **dense repair localized by drafter
uncertainty**, not sparse boundary detection; the offline trigger<=15-20%
pass bars were fixed-budget replay artifacts and do not bind online (graft
adds branches without removing EAGLE nodes, so MAT can only gain; the cost
is verify tokens).

Profile (t=0.90): fusion_logic 1.37% + draft_sam 0.03% explicit overhead;
verify 72.97% dominant. Per-step time +2.8-3.4% vs baseline = capture/tolist
tax (inside draft_eagle) + Python tree parse + verify growth on triggered
steps.

## Conclusions

1. Online Drafter-MARS (graft repair, theta=0.86) beats SAM[EAGLE3] on both
   MAT (+4.01%) and throughput (+1.15%) with statistical separation.
2. Graft repair at the triggering parent is the right action; root-anchored
   fuse underperforms (user observation motivating S3) and always-on
   underperforms (control above).
3. Remaining throughput headroom is engineering: gate/capture tensorization
   (~1-2% expected) and graft budget tuning; verify growth is productive
   cost.

## Open items

- Budget sweep `boundary_graft_max_sam_nodes in {4,6,8,12}` at theta=0.86.
- Gate/capture tensorization (outputs must stay bit-identical).
- naive_fuse ablation arm at 0.86 (optional; earlier evidence already favors
  graft).
- Offline trace-enrichment raw-pair one-level shift (design.md open issue)
  still unresolved; offline theta grids suspect, online sweep supersedes them
  for operating-point choice.
