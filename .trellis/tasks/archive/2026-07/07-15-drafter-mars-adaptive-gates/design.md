# Design: Adaptive and learned gates for drafter-MARS

Depends on: prd.md. Baseline semantics = commit `fbd338c` drafter_mars branch.

Closeout: Phase A is retained as experimental/ablation code. Phase B was
canceled by user decision on 2026-07-17 because the final throughput gain was
too small to justify learned-gate complexity and cross-domain GPU work.

## Deletability contract

New modules own all logic; hook points are guarded one-liners:

```text
samd/fusion/drafter_mars_adaptive.py   # Phase A: controller + budget + multi-trigger (pure, no torch)
samd/fusion/drafter_mars_prune.py      # Phase A3: fixed-budget EAGLE leaf pruning (pure, no torch)
samd/fusion/gate_head.py               # Phase B: online head inference (torch, lazy import)
evaluation/gate_head/                  # Phase B: capture/dataset/train/eval (offline)
tests/test_drafter_mars_adaptive.py    # torch-free
tests/test_gate_head.py                # torch-gated
```

Hooks limited to: `SamdConfig` fields + validation block, `inference_samd.py`
CLI flags, and inside the existing `fusion_mode == "drafter_mars"` branch of
`samd/utils.py` (theta source, budget source, graft loop, gate dispatch).
All defaults off → identical control flow to today (single graft, fixed theta,
fixed budget, ratio gate).

## Phase A — adaptive trio

### A1. Online theta self-calibration (`AdaptiveThetaController`)

Per-request integral controller on the trigger indicator (steps where the
match-length gate did NOT fire, i.e. steps where the ratio gate ran):

```text
state: theta (init = drafter_mars_theta), ema (init = target)
update(triggered):
    ema   = (1 - m) * ema + m * 1[triggered]
    theta = clip(theta + eta * (ema - target), theta_min, theta_max)
```

- Direction: observed rate above target → theta rises (stricter).
- Config: `drafter_mars_adaptive_theta: bool = False`,
  `drafter_mars_target_trigger_rate: float = 0.75`,
  `drafter_mars_theta_step: float = 0.02` (eta), momentum m = 0.05 fixed,
  clip [0.50, 0.995].
- Lifecycle: instance stored on `DraftModel`, recreated in `reset()` (per
  request; no cross-request leakage). utils.py reads
  `draft.drafter_mars_controller` when the switch is on, else config theta.
- Metadata: per-step `theta_effective`, controller `ema`.

### A2. Depth/ratio-aware budget (`resolve_graft_budget`)

```text
mode "fixed": base                                  (default, current behavior)
mode "depth": max(min_nodes, base - trigger_depth)  # deeper → shorter
mode "ratio": max(min_nodes, round(base * (ratio - theta) / (1 - theta) clipped to [0,1] * 1.0 + min share))
```

Pure function `resolve_graft_budget(mode, base, depth, ratio, theta, min_nodes=2)`.
Config: `drafter_mars_budget_mode: Literal["fixed","depth","ratio"] = "fixed"`.
`base` = `boundary_graft_max_sam_nodes` (unchanged).

### A3. Multi-point graft

- Gate module gains `all_top_path_triggers(...) -> List[TriggerInfo]`
  (ascending depth; reuses the same greedy-path walk as
  `earliest_top_path_trigger`, which stays as the K=1 fast path).
- Config: `drafter_mars_max_grafts: int = 1`,
  `drafter_mars_total_graft_nodes: int = 16` (global cap across grafts).
- utils.py loop (only when max_grafts > 1): for each trigger in depth order,
  up to K grafts and the global node cap — walk that parent's prefix
  (stateless transfer_state fold, as today), gen_draft_raw with its
  per-graft budget from A2, `graft_sam_at_depth` at that parent. Original
  node indices are stable because grafts only append nodes, so sequential
  grafts at original parent indices remain valid. Empty continuations skip
  without consuming a graft slot.
- Metadata: `graft_count`, per-graft (depth, ratio, nodes_added).

### Interactions

A1 changes only theta input; A2 only budget input; A3 only the repair loop.
They compose freely; eval arms test each alone plus the full trio.

## Phase A3 — fixed verifier-node budget (2026-07-16 pivot)

Composite experiments raised MAT by 5.65--6.38%, but wall-clock throughput by
only 1.5--2.0% because the appended graft nodes increased verifier cost per
step. Fixed-budget repair therefore changes the integration rule rather than
the gate:

- `drafter_mars_tree_budget=None` preserves append-only behavior.
- A positive budget is root-inclusive; the primary arm uses 60, equal to the
  stock EAGLE3 tree size.
- After graft/extension, keep every newly added SAM node, its ancestor closure,
  and the EAGLE greedy top path.
- Iteratively remove only unprotected EAGLE leaves with the lowest mean path
  logprob, remapping the surviving tree to valid contiguous indices.
- Apply the same helper to single graft, multi-graft, subtree repair, and leaf
  extension so the verifier-node contract is consistent across mechanisms.

The first evaluation compares append-only r8/e16 against budget-60 r8/e16 and
budget-60 r8/K2/e16. A budget violation is a hard error rather than a silent
fallback.

## Phase B — learned multi-task decision head (revised 2026-07-15 after A8)

Positioning (user decision): the learned head is the METHOD; MARS ratio is
the zero-shot baseline; ACI-framed adaptive theta is the training-free
calibration baseline; fixed theta is the ablation. Phase A data justifies
this: hand-crafted budget schedules LOST MAT (depth -1.99%, ratio -2.32%)
while more/multi repair won (K=2 +0.67%, b12 +0.27% over b8) — where and how
much to repair should be predicted from data, not ruled.

### Head

One shared MLP trunk (features → 2×64 ReLU) with two outputs per top-path
parent:

1. `p_trigger` — probability the verifier rejects at this parent (BCE).
2. `budget` — how many SAM nodes to graft here, 4-class over {0, 4, 8, 12}
   (class CE; 0 = not worth grafting even if triggered — subsumes precision).

Online selection: score all top-path parents in one batched forward, take
top-K by `p_trigger` above tau (K = `drafter_mars_max_grafts`, reuse Phase A
multi-graft machinery + `drafter_mars_total_graft_nodes` cap), per-graft
budget = head's class. tau calibrated on val split; optionally ACI-adjusted
online (combines Phase A controller with the learned score — principled and
mainstream).

### Features (per top-path parent; all available online, no extra capture cost)

ratio, z1, z2, greedy-child local_logprob, sibling logprob delta, parent
depth (/ eagle3_depth), cumulative path logprob, normalized path logprob,
step best_match (match-length gate residual), children count of parent.
Optional (off by default): parent-node drafter hidden state (eagle3
`out_hidden`, projected 4096→64 inside the head) — only if feature-only AUC
is unsatisfying.

### Labels (two capture streams)

- **Trigger stream** (from a plain drafter_mars run with capture on): per
  step where the EAGLE draft was used, compare greedy top path with the
  verifier-accepted path. Parents strictly above the divergence depth →
  label 0; the parent AT the divergence → label 1; parents below → masked
  (verifier never saw them).
- **Budget stream** (from a K=2/b12 drafter_mars run with capture on): per
  executed graft, count how many of its grafted nodes the verifier accepted
  (accepted candidate row ∩ graft node range) → budget class label at that
  parent's features. This is direct supervision for "how much repair pays".

### Capture / training / serving

- `--gate-head-capture <path>`: JSONL, one record per step — features per
  top-path parent, accepted path, graft node ranges + accepted counts when
  grafts happened. Implemented inside the drafter_mars branch (guarded, zero
  cost when off).
- `evaluation/gate_head/dataset.py`: streams → (X, y_trigger, y_budget,
  masks); split by question id (train 0-99 / val 100-131 / test 132-164).
- `evaluation/gate_head/train.py`: standardize, multi-task loss
  (BCE + CE, equal weight to start), report trigger AUC + budget accuracy
  vs the ratio-threshold baseline on the same split; save state_dict +
  feature-spec/normalization JSON.
- `samd/fusion/gate_head.py`: lazy torch module, loads checkpoint once,
  batched forward per step. Config: `drafter_mars_gate_kind:
  Literal["ratio","learned"] = "ratio"`, `drafter_mars_gate_head_path`,
  `drafter_mars_gate_tau: float = 0.5`. Missing checkpoint with
  gate_kind="learned" → hard error (no silent fallback).

## Eval plan (server batches, same pattern as 06-06)

- Batch A (DONE 2026-07-15): ref equivalence OK (MAT 7.7849). K=2 → MAT
  7.8372 (+4.71% vs SAM[EAGLE3] baseline) at equal tok/s — new operating
  point. K=3 marginal. Budget schedules depth/ratio and trio hurt (rule
  schedules falsified). Adaptive theta in-domain -0.35% (cross-domain arm
  deferred).
- Batch B: capture runs (trigger stream + budget stream) → train → held-out
  AUC/accuracy report → online arms: ratio-K2 (reference) vs learned-K2 vs
  learned-K2-with-learned-budget. MAT + tok/s + trigger rate each.

## Risks

| Risk | Mitigation |
| --- | --- |
| Adaptive theta oscillates / collapses to clip bound | small eta, clip range, per-step metadata to diagnose; target-rate arm compared against fixed arm |
| Multi-graft inflates verify cost faster than MAT | global node cap + per-graft budgets; K sweep |
| Hidden-state capture too slow/large | optional flag; feature-only head is the default path |
| Label noise (divergence ≠ drafter fault) | boundary-parent labeling only on steps where EAGLE draft was used; masked below-divergence parents; report AUC before any online claim |
| Budget labels only exist at triggered sites (selection bias) | budget stream from the dense K=2 run (75%+ trigger rate) covers most steps; report label coverage |
| Head overfits HumanEval | question-id split now; cross-domain training data is the follow-up task |
