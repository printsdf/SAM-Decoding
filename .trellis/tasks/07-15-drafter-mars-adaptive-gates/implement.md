# Implement: Adaptive and learned gates for drafter-MARS

Depends on: `design.md`. Order: Phase A fully lands and is measured on the
server before Phase B starts (user decision 2026-07-15).

## Checklist

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
- [ ] A8 Server run + record results in task docs.

### Phase B — learned gate head (start after A8 readout)

- [ ] B1 Capture: `--gate-head-capture` flag → per-step JSONL (features per
      top-path parent + acceptance path); no overhead when off.
- [ ] B2 `evaluation/gate_head/dataset.py`: JSONL → labeled matrix
      (boundary-parent labeling per design), question-id split.
- [ ] B3 `evaluation/gate_head/train.py`: MLP train/eval, AUC vs ratio
      baseline on held-out, checkpoint + feature-spec JSON.
- [ ] B4 `samd/fusion/gate_head.py` online inference +
      `drafter_mars_gate_kind/gate_head_path/gate_tau` config + CLI.
- [ ] B5 Tests: dataset labeling on synthetic fixtures (torch-free where
      possible); head forward shape/threshold (torch-gated).
- [ ] B6 Server: capture q0-99/train, held-out AUC report, online arm vs
      ratio arm; record results.

## Validation commands

```bash
python3 -m py_compile samd/fusion/drafter_mars_adaptive.py samd/fusion/drafter_mars_gate.py samd/samd_config.py samd/utils.py samd/draft.py evaluation/inference_samd.py
python3 -m pytest tests/test_drafter_mars_adaptive.py tests/test_drafter_mars_gate.py -q
bash -n scripts/run_drafter_mars_adaptive_batch.sh
# server: pytest (torch-gated) + batch script; default-off arm must reproduce MAT 7.7849
```

## Rollback

- Phase A: revert hook commit(s) + delete `drafter_mars_adaptive.py` and its
  tests; defaults-off equivalence means intermediate states stay safe.
- Phase B: delete `gate_head.py` + `evaluation/gate_head/` + flags.
