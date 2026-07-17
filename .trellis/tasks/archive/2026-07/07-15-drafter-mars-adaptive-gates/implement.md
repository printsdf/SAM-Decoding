# Implement: Adaptive and learned gates for drafter-MARS

Depends on: `design.md`. Phase A landed and was measured; Phase B was later
canceled during project closeout (user decision 2026-07-17).

## Checklist

### Phase A2 — author-horizon grafts + leaf extension (user decision 2026-07-15)

- [x] A2.1 Grafts use the author's SAM semantics: `gen_draft_raw` with the
      `n_predicts` horizon (no artificial caps); `TreeSpec.graft_sequence_at`
      (walk-reuse-or-append, generalizing the author's `graft_sequence`)
      replaces the boundary_graft chain append.
- [x] A2.2 `drafter_mars_extend`: on non-triggered / repair-unavailable steps,
      graft the SAM continuation at the greedy leaf (n_predicts horizon).
- [x] A2.3 Deleted previous-experiment code: `samd/fusion/boundary_graft.py`,
      `samd/fusion/rejection_boundary.py`, fusion_mode branches
      naive/rejection_boundary/boundary_graft in utils.py, boundary/budget
      config fields + CLI flags, budget schedules in the adaptive module.
      fusion_mode is now `none | drafter_mars`.
- [x] A2.4 `max_predicts` auto-raised for drafter_mars worst-case tree size
      (fixes the pre-existing K2=77 > 70 guard hole).
- [x] A2.5 Batch `scripts/run_drafter_mars_horizon_batch.sh` (same-boot
      baseline, g40, g40_k2, ext, g40_k2_ext); stale batch scripts removed.
      NOTE: graft length semantics changed (8 → n_predicts), so Phase A b8
      numbers are historical; horizon batch re-baselines everything.
- [x] A2.6 Server run + record results (4 batches; operating point r8k2_e16 =
      MAT +6.28% / +2.0% tok/s; see
      `docs/experiments/results/2026-07-16-horizons-extension-oracle.md`).

### Phase A — adaptive trio

- [x] A1 `samd/fusion/drafter_mars_adaptive.py`: `AdaptiveThetaController`
      (integral controller, clip, ema), `resolve_graft_budget` (fixed/depth/
      ratio), pure + torch-free.
- [x] A2 Gate module: `all_top_path_triggers` returning every triggering
      top-path parent in depth order (K=1 path keeps using
      `earliest_top_path_trigger`).
- [x] A3 `SamdConfig` + validation + CLI: `drafter_mars_adaptive_theta`,
      `drafter_mars_target_trigger_rate`, `drafter_mars_theta_step`,
      `drafter_mars_budget_mode`, `drafter_mars_max_grafts`,
      `drafter_mars_total_graft_nodes`.
- [x] A4 `DraftModel.reset()`: controller lifecycle (created only when
      adaptive switch on).
- [x] A5 `samd/utils.py` drafter_mars branch hooks: theta source, budget
      source, multi-graft loop (guarded: max_grafts == 1 → exact current
      single-graft path); per-step metadata (theta_effective, graft_count,
      per-graft depth/ratio/nodes).
- [x] A6 Tests `tests/test_drafter_mars_adaptive.py` (torch-free): controller
      direction + clip + convergence toward target on synthetic streams;
      budget modes (fixed identity, depth monotone, ratio monotone, min
      floor); multi-trigger ordering + cap semantics; torch-gated config
      validation additions.
- [x] A7 py_compile + torch-free pytest; batch script
      `scripts/run_drafter_mars_adaptive_batch.sh` (reference arm, adaptive
      arm, budget arms, K arms, trio arm, default-off equivalence assert,
      speed.py summary, Feishu + shutdown).
- [x] A8 Server run + record results in task docs (batch 2026-07-15: K=2 new
      operating point, budget schedules falsified; see results note).

### Phase A3 — fixed verifier-node budget

- [x] A3.1 Pure leaf-pruning module: protect SAM nodes/ancestors + EAGLE greedy
      path; prune lowest mean-path-logprob EAGLE leaves; remap valid tree.
- [x] A3.2 Optional root-inclusive `drafter_mars_tree_budget` config/CLI,
      default off; metadata records pre-prune/final/pruned node counts.
- [x] A3.3 Apply uniformly to graft K1/K>1, subtree repair, and extension.
- [x] A3.4 Torch-free tests and fixed-budget batch script.
- [x] A3.5 Server batch: append r8/e16 vs budget60 r8/e16 vs budget60 K2.
      Current mean-path/protect-all-SAM pruning is negative; see
      `docs/experiments/results/2026-07-16-fixed-verifier-budget.md`.
- [x] A3.6 Preallocated-budget batch script: stock E60 baseline, append
      r8/e16, E53+r8 repair-only, E53+r8/e8, and a profiled E53+r8/e8 arm.
- [x] A3.7 Server preallocated-budget batch + result record. E53+r8/e8 is
      wall-time neutral to append but MAT-dominated; see
      `docs/experiments/results/2026-07-17-preallocated-budget.md`.

### Phase B — learned multi-task decision head (canceled by user 2026-07-17)

- [ ] CANCELED B1 Capture: `--gate-head-capture <path>` → per-step JSONL with
      per-top-path-parent features, accepted path, graft node ranges +
      accepted counts (trigger stream from plain run; budget stream from
      K=2/b12 run); zero overhead when off.
- [ ] CANCELED B2 `evaluation/gate_head/dataset.py`: streams → (X, y_trigger,
      y_budget, masks); boundary labeling with below-divergence masking;
      question-id split 0-99/100-131/132-164.
- [ ] CANCELED B3 `evaluation/gate_head/train.py`: shared-trunk multi-task MLP
      (BCE trigger + CE budget over {0,4,8,12}), held-out trigger AUC +
      budget accuracy vs ratio-threshold baseline, checkpoint +
      feature-spec/normalization JSON.
- [ ] CANCELED B4 `samd/fusion/gate_head.py` online: batched per-step forward, top-K
      selection above tau, per-graft learned budget (reuses Phase A
      multi-graft machinery + total cap); config/CLI
      `drafter_mars_gate_kind/gate_head_path/gate_tau`; missing checkpoint
      with gate_kind="learned" → hard error.
- [ ] CANCELED B5 Tests: dataset labeling + masking on synthetic fixtures
      (torch-free); head forward shape/threshold/top-K (torch-gated).
- [ ] CANCELED B6 Server: capture runs → train → AUC report → online arms ratio-K2 vs
      learned-K2 vs learned-K2+learned-budget; record results.

## Validation commands

```bash
python3 -m py_compile samd/fusion/drafter_mars_adaptive.py samd/fusion/drafter_mars_gate.py samd/fusion/drafter_mars_prune.py samd/samd_config.py samd/utils.py samd/draft.py evaluation/inference_samd.py
python3 -m pytest tests/test_subtree_graft.py tests/test_drafter_mars_adaptive.py tests/test_drafter_mars_gate.py -q
bash -n scripts/run_drafter_mars_preallocated_budget_batch.sh
# server: pytest (torch-gated) + batch script; default-off arm must reproduce MAT 7.7849
```

## Rollback

- Phase A: revert hook commit(s) + delete `drafter_mars_adaptive.py` and its
  tests; defaults-off equivalence means intermediate states stay safe.
- Phase B: delete `gate_head.py` + `evaluation/gate_head/` + flags.
