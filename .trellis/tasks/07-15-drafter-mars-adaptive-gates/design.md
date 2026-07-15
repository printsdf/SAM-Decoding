# Design: Adaptive and learned gates for drafter-MARS

Depends on: prd.md. Baseline semantics = commit `fbd338c` drafter_mars branch.

## Deletability contract

New modules own all logic; hook points are guarded one-liners:

```text
samd/fusion/drafter_mars_adaptive.py   # Phase A: controller + budget + multi-trigger (pure, no torch)
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

## Phase B — learned gate head (sketch; refine before B starts)

- **Features** per top-path parent: ratio, depth, local_logprob(top1/top2),
  logprob delta, cumulative/normalized path logprob, sam_match hint
  (best_match at step start); optional drafter hidden state of the parent
  node (eagle3 `out_hidden`, capture behind a flag — heavy, off by default).
- **Labels**: from the verifier — greedy top path vs accepted path; the
  shallowest top-path parent at or below which the verifier diverged is the
  positive boundary; parents strictly above it are negatives.
- **Capture**: `--gate-head-capture <path>` on inference_samd; writes JSONL
  (one record per step: features per parent + accept path). Reuses the
  existing profiler step-metadata plumbing where possible.
- **Trainer**: `evaluation/gate_head/train.py` — standardize features, MLP
  (2×64, ReLU, BCE), split by question id, report AUC vs the ratio-threshold
  baseline on the same split; save TorchScript-able checkpoint + feature
  spec JSON.
- **Online**: `drafter_mars_gate_kind: Literal["ratio","learned"] = "ratio"`,
  `drafter_mars_gate_head_path: Optional[str]`. `samd/fusion/gate_head.py`
  loads the checkpoint once, scores top-path parents in one small matmul,
  triggers when p > tau (`drafter_mars_gate_tau: float = 0.5`); missing
  checkpoint → hard error (no silent fallback), ratio gate stays the
  explicit baseline arm.

## Eval plan (server batches, same pattern as 06-06)

- Batch A: fixed t086/b8 reference + adaptive-theta arm (target 0.75) +
  budget-mode arms (depth, ratio) + multi-graft arms (K=2, K=3) + full trio.
  Plus default-off equivalence check (MAT 7.7849).
- Batch B: capture run (0-99 train / 100-164 held-out), train, AUC report,
  online learned-gate arm vs ratio arm.

## Risks

| Risk | Mitigation |
| --- | --- |
| Adaptive theta oscillates / collapses to clip bound | small eta, clip range, per-step metadata to diagnose; target-rate arm compared against fixed arm |
| Multi-graft inflates verify cost faster than MAT | global node cap + per-graft budgets; K sweep |
| Hidden-state capture too slow/large | optional flag; feature-only head is the default path |
| Label noise (divergence ≠ drafter fault) | boundary-parent labeling only on steps where EAGLE draft was used; report AUC before any online claim |
