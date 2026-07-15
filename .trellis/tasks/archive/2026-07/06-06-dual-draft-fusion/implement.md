# Implement: Drafter-MARS modularization + cleanup

Depends on: `design.md` (A + three packages)

## Checklist

### Phase S4 — Hot-path tensorization (session 2026-07-15, outputs must stay bit-identical)

- [x] S4.1 `eagle3_parents_from_buffers` torch helper (replaces per-step O(n^2) Python mask parse); drafter_mars gate runs on the parents list, TreeSpec built only on triggered graft steps.
- [x] S4.2 Raw-pair capture gathers `headout` at the existing log-softmax topk indices instead of a fresh vocab-wide topk per depth.
- [x] S4.3 Torch-gated equivalence tests `tests/test_eagle3_parents.py` (parents vs `from_eagle3_buffers`; gather vs vocab topk). Server acceptance: rerun t086 → MAT must equal 7.7849 exactly.

### Phase S3 — Graft repair for drafter_mars (session 2026-07-14, user-approved)

- [x] S3.1 Gate: return earliest triggering top-path parent (node index + depth + ratio); keep bool/max-ratio compat.
- [x] S3.2 `SamdConfig`: `drafter_mars_repair` Literal["graft","naive_fuse"] default "graft" + validation; CLI `--drafter_mars_repair`.
- [x] S3.3 `samd/utils.py`: triggered branch → graft path (prefix transfer_state → SAM continuation → `graft_sam_at_depth` at triggering parent, budget `boundary_graft_max_sam_nodes`); `naive_fuse` keeps old behavior.
- [x] S3.4 Metadata/stats: record repair kind, trigger depth/ratio, sam nodes added.
- [x] S3.5 Tests: earliest-parent selection; repair-mode config validation; existing gate tests still pass.
- [x] S3.6 py_compile + torch-free pytest; update eval commands note.

### Phase S2 — Online `fusion_mode="drafter_mars"` (session 2026-07-14)

- [x] S2.1 `SamdConfig`: add `"drafter_mars"` to `fusion_mode` literal + validation; add `drafter_mars_theta` (default 0.90), `drafter_mars_variant` ("top_path"); map `fusion_config` like `rejection_boundary`.
- [x] S2.2 New `samd/fusion/drafter_mars_gate.py`: pure gate functions (greedy top-path parents, z2/(z1+1e-10) ratios, trigger decision).
- [x] S2.3 `samd/utils.py` `gen_candidates`: `drafter_mars` branch per design data flow (match-length gate first → EAGLE3 raw-logit draft → ratio gate → `fuse_eagle_sam_naive` or pure EAGLE3).
- [x] S2.4 `samd/draft.py`: include `drafter_mars` in fusion-stats enabled modes.
- [x] S2.5 `evaluation/inference_samd.py`: CLI `--fusion_mode drafter_mars`, `--drafter_mars_theta`.
- [x] S2.6 Tests `tests/test_drafter_mars_gate.py`: ratio direction, top-path filtering (off-path high ratio ignored), stricter theta → fewer triggers, config validation for all fusion_mode values.
- [x] S2.7 Validation: py_compile + pytest (gate tests + existing offline suite) + record baseline/drafter_mars eval commands.

### Phase M1 — Create packages (no deletes yet)

- [x] M1.1 Create `evaluation/oracle/{__init__,types,load,select,analyze,cli,rejection_boundary,depth_decoupled}.py` by splitting current oracle modules.
- [x] M1.2 Create `evaluation/boundary/{__init__,types,candidates,metrics,selection,sweep,low_margin,calibrate,cli}.py` from `analyze_boundary_predictor.py` with private APIs promoted.
- [x] M1.3 Create `evaluation/drafter_mars/{__init__,parents,predictors,schema,calibrate}.py` from `drafter_mars_predictor.py`; import only public boundary/oracle APIs.
- [x] M1.4 Replace old top-level modules with thin re-export shims.
- [x] M1.5 `py_compile` + import smoke for the three packages.

### Phase M2 — Retarget consumers

- [x] M2.1 Update tests to prefer new package imports.
- [x] M2.2 Update `scripts/profile_boundary_predictor_humaneval.sh` to `python -m evaluation.oracle.cli` / `evaluation.oracle.rejection_boundary` / `evaluation.boundary.cli`.
- [x] M2.3 Update any kept script that still calls old paths (or leave on shims).
- [x] M2.4 Run available unit tests / synthetic calibrate. Offline suite: 25 passed.

### Phase M3 — Delete unrelated

- [x] M3.1 Dry-run list of evaluation deletes; confirm `inference_samd` + prep stay.
- [x] M3.2 Delete evaluation unrelated sources + `*.bak` + `constrained_tasks/` (+ `model/` if unused by kept inference).
- [x] M3.3 Delete out-of-scope scripts (MT-Bench/MedQA/overhead/eagle1-2/pld/token_recycle/sam-only/boundary_graft/vmiss/equal/speed).
- [x] M3.4 Residuals: `.DS_Store`, `__pycache__`, `.worktrees/...`, `git worktree prune`.
- [x] M3.5 Grep kept scripts for dangling refs to deleted files.

### Phase M4 — Docs touch

- [x] M4.1 Update task README current state (implemented + modularized; smoke pending).
- [x] M4.2 Update experiments README if needed.
- [x] M4.3 Note in session report that modularization landed.

## Validation commands

```bash
python3 -m py_compile \
  evaluation/oracle/*.py \
  evaluation/boundary/*.py \
  evaluation/drafter_mars/*.py \
  evaluation/oracle_fusion_analysis.py \
  evaluation/analyze_boundary_predictor.py \
  evaluation/drafter_mars_predictor.py \
  evaluation/inference_samd.py

python3 -c "from evaluation.drafter_mars import calibrate_drafter_mars, DRAFT_MARS_THETA_GRID; from evaluation.boundary import evaluate_predictions, Prediction; from evaluation.oracle import load_decode_traces, select_eagle3_only; print('ok')"

bash -n scripts/profile_boundary_predictor_humaneval.sh

# if pytest available:
python3 -m pytest tests/test_boundary_predictor_analysis.py tests/test_oracle_fusion_analysis.py tests/test_naive_fusion.py tests/test_oracle_depth_decoupled.py -q
```

## Rollback

- Revert the modularization commit(s).
- Shims mean intermediate states should still import via old paths.
- Do not delete samd runtime in any step.

## Delete inventory (M3)

### evaluation/ delete

```
analyze_linear_alignment.py
analyze_vmiss.py
equal.py
speed.py
eval.py
eval_standard.py
eval_llama3.py
eval_vicuna.py
eval_naive_fusion.py
inference_baseline.py
inference_eagle.py
inference_eagle2.py
inference_pld.py
inference_sam_only.py
inference_sam_only.py.bak
inference_token_recycle.py
medqa_prep.py
medquad_prep.py
profile_entry.py
profile_sam_only.py
profile_samd.py
oracle_high_precision_sam.py
constrained_tasks/
model/                    # only if inference_samd does not import it
*.bak under evaluation/
```

### evaluation/ keep

```
oracle/  boundary/  drafter_mars/
inference_samd.py
humaneval_prep.py
oracle_fusion_analysis.py          # shim
analyze_boundary_predictor.py      # shim
drafter_mars_predictor.py          # shim
oracle_rejection_boundary.py       # shim
oracle_depth_decoupled.py          # shim (or only package path)
data/                              # gitignored; leave alone
```

### scripts/ delete (out of scope)

```
check_eagle3_equiv.sh
equal.sh
eval_boundary_graft_sweep.sh
eval_boundary_graft.sh
eval_eagle_only_humaneval_full.sh
eval_rejection_boundary_humaneval_full.sh
eval_rejection_boundary_phase1.sh
inference_baseline.sh
inference_eagle.sh
inference_eagle2.sh
inference_pld.sh
inference_samd_eagle3.sh
inference_samd_sam_only.sh
inference_samd.sh
inference_token_recycle.sh
profile_fusion_humaneval.sh
profile_fusion_medqa.sh
profile_fusion_mt_bench.sh
profile_fusion_overhead_no_tail.sh
profile_fusion_overhead.sh
profile_naive_fusion.py
run_bench_cross_domain_speedup.sh
run_medqa_sam_graft_compare.sh
run_medqa_vmiss_eval.sh
run_medquad_sam_graft_compare.sh
run_profile.sh
speed.sh
test_medqa_vmiss_smoke.sh
test_samd_eagle3.sh
test_samd_sam_only.sh
test_samd.sh
```

### scripts/ keep

```
profile_boundary_predictor_humaneval.sh
run_canonical_smoke_q0_20.sh   # if still useful for smoke
```

Verify `run_canonical_smoke_q0_20.sh` before keep/delete.
