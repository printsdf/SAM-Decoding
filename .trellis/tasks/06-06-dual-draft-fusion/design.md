# Design (Stage 2): Online `fusion_mode="drafter_mars"`

Status: in progress (session 2026-07-14)
Scope: samd runtime online gate + CLI + tests. Offline packages untouched (read-only reference).

## Baseline definition (verified in code)

SAM[EAGLE3] baseline = `tree_method=eagle3, fusion_mode=none, tree_fusion=none`.
Per decode step, `samd/draft.py` `DraftModel.lookup` (line ~439):
`max(match_dyn, match_static - len_bias) >= len_threshold` (**`>=`**, default 5)
→ SAM sequence draft; else → pure EAGLE3 tree draft. This is the system
baseline command target, not `fusion_mode=naive` (which builds a fused tree).

## Online Drafter-MARS data flow

Per decode step, in `gen_candidates` (`samd/utils.py`), new branch
`fusion_mode == "drafter_mars"`:

1. SAM lookup (dyn + static, `len_bias` applied) → `best_match`.
2. `best_match >= len_threshold` → SAM sequence draft (identical to baseline;
   ratio gate never overrides a match-length hit).
3. Else → EAGLE3 draft with `return_logprobs=True, return_raw_logits=True`
   (raw-pair extraction is a cheap top-2 gather; same capture path as profiling).
4. Compute the greedy top path online: from root, repeatedly descend to the
   child with max `local_logprob`. Offline `is_top_path` used the verifier
   acceptance path, which does not exist pre-verification — the greedy
   drafter top path is the online proxy.
5. For each parent node on that path with raw pair `(z1, z2)`:
   `ratio = z2 / (z1 + 1e-10)`; trigger when `ratio > drafter_mars_theta`.
6. Triggered and SAM has a non-empty continuation → reuse
   `fuse_eagle_sam_naive` (same fuse path as `fusion_mode=naive`) to graft the
   SAM sequence into the EAGLE tree. Not triggered → pure EAGLE3 tree
   (identical to baseline).

Gate logic lives in `samd/fusion/drafter_mars_gate.py` as pure functions
(top-path extraction from TreeSpec + logprobs, ratio computation, trigger
decision) so it is unit-testable without a model.

## Config / CLI

- `SamdConfig.fusion_mode` gains `"drafter_mars"`; requires `tree_method=eagle3`.
- `drafter_mars_theta: float = 0.90` (validated in (0, +inf), grid-compatible).
- `drafter_mars_variant: Literal["top_path"] = "top_path"` (reachable deferred).
- `fusion_config` maps `drafter_mars` → naive fuse config (like `rejection_boundary`).
- `evaluation/inference_samd.py`: `--fusion_mode drafter_mars`,
  `--drafter_mars_theta`.
- Fusion stats: mode added to the enabled set; per-step metadata records
  `ratio_triggered`, max top-path ratio, theta.

## Open issue: offline trace enrichment raw-pair indexing (found 2026-07-14)

Verified capture convention in `eagle3_model.topK_genrate`: `raw_pairs[i]` holds
the (z1, z2) of the expansion that PRODUCED node i (child-index convention;
root row is a NaN placeholder). The online gate reads pairs at the greedy
child, which is correct. But offline `naive_fusion._eagle_trace_metadata`
(~line 220) reads `raw_logits[parent_index]` — under this convention that is
the GRANDPARENT's expansion pair, and depth-1 candidates get None. Suspected
one-level shift in all Stage-1 offline ratio calibration (theta grid results
possibly computed on shifted ratios). Not fixed in this session (would
invalidate Stage-1 numbers); needs a user decision + recalibration if
confirmed.

## Revision (2026-07-14, user decision): graft repair replaces naive fuse

User observation: naive fusion underperforms graft; the MARS signal localizes
the uncertain parent, so repair must happen AT that node. Revised trigger
action:

1. Gate unchanged (greedy top-path parents, ratio `> theta`, match-length gate
   first), but it now selects the **earliest** (shallowest, tie → lowest tree
   index) triggering parent, matching offline `predict_draft_mars_top_path`.
2. Repair reuses the `boundary_graft` machinery
   (`samd/fusion/boundary_graft.py`): extract the EAGLE top-path prefix up to
   the triggering parent, `transfer_state` it into SAM so the continuation is
   conditioned on the drafted prefix, then `graft_sam_at_depth` at that parent
   (budget `boundary_graft_max_sam_nodes`, default 8). EAGLE tree preserved,
   no truncation.
3. `drafter_mars_repair: Literal["graft", "naive_fuse"] = "graft"` keeps the
   old root-anchored fuse as an ablation arm under the same gate; CLI
   `--drafter_mars_repair`.

Rationale: signal-action alignment (repair at the located boundary), zero
EAGLE truncation cost, and consistency with the Stage-1 oracle ceiling which
is graft-semantics (`rejection_boundary` oracle).

## Eval commands (HumanEval full 0–164, profile off)

Baseline:
```bash
python3 evaluation/inference_samd.py --model-type llama3 --template llama3 \
  --model-path <target> --tree_model_path <eagle3> --model-id samd_eagle3_baseline \
  --bench-name humaneval --question-begin 0 --question-end 164 \
  --tree_method eagle3 --fusion_mode none --tree_fusion none \
  --samd_len_threshold 5 --samd_len_bias 5
```
Drafter-MARS: same command with `--fusion_mode drafter_mars --drafter_mars_theta 0.90 --drafter_mars_repair graft`
and `--model-id samd_eagle3_drafter_mars_t090`. Use `--drafter_mars_repair naive_fuse`
for the root-anchored fuse ablation arm. Compare MAT (accept length per
step) and tokens/s from the run summary.

---

# Design (Stage 1): Drafter-MARS offline modularization + cleanup

Status: approved direction (user 2026-07-13)
Scope: **A** — offline evaluation side only; **samd runtime untouched**.
Layout: **three packages** under `evaluation/`.

## Goal

1. Make the Drafter-MARS offline path a clean layered library (no private-API imports).
2. Delete evaluation/scripts/residuals that are not needed for Drafter-MARS Stage-1 offline evidence.
3. Keep smoke entry (`scripts/profile_boundary_predictor_humaneval.sh`) and samd runtime working.

## Non-goals

- No samd/EAGLE3/SAM runtime refactor (capture path stays as-is).
- No online `fusion_mode` change.
- No experiment re-run in this change.
- No broad rewrite of archived task docs (README/status notes can get a short touch).

## Package layout

```text
evaluation/
  oracle/                         # same-trace oracle types + load + select + MAT
    __init__.py                   # public re-exports
    types.py                      # OracleCandidate, DecodeStep, MethodResult
    load.py                       # load_decode_traces, coerce_decode_steps, helpers
    select.py                     # select_*, selected_mat, candidate_* helpers
    analyze.py                    # analyze_steps, render_table (depth_decoupled optional)
    rejection_boundary.py         # rejection-boundary oracle CLI/library
    depth_decoupled.py            # depth-stratified oracle (baseline table)
    cli.py                        # `python -m evaluation.oracle.cli`
  boundary/                       # shared boundary-predictor scaffolding
    __init__.py
    types.py                      # Prediction, DEFAULT_NODE_BUDGET, bars
    candidates.py                 # eagle/sam candidate accessors (public)
    metrics.py                    # evaluate_predictions, prefix/depth hits, sam rescue
    selection.py                  # constrained selection + Q2 guard (public)
    sweep.py                      # threshold grid + run_threshold_sweep
    low_margin.py                 # unconditioned low_margin baseline
    calibrate.py                  # default-suite calibrate()
    cli.py                        # `python -m evaluation.boundary.cli`
  drafter_mars/                   # Drafter-MARS suite only
    __init__.py
    parents.py                    # ParentRecord + parent reconstruction
    predictors.py                 # top_path / reachable / delta predictors
    schema.py                     # schema_report, raw_logit_diagnostics, require_*
    calibrate.py                  # calibrate_drafter_mars
  # thin legacy shims (optional, temporary) OR deleted after script updates:
  # oracle_fusion_analysis.py -> re-export / -m shim
  # analyze_boundary_predictor.py -> re-export / -m shim
  # drafter_mars_predictor.py -> re-export
  # oracle_rejection_boundary.py -> re-export
  # oracle_depth_decoupled.py -> re-export

  inference_samd.py               # KEEP — smoke profiling entry
  humaneval_prep.py               # KEEP — question file prep
  ...deleted unrelated...
```

### Dependency rule (hard)

```text
drafter_mars  -->  boundary  -->  oracle
drafter_mars  -->  oracle
boundary      -X-> drafter_mars   (boundary may lazy-import drafter_mars only in CLI/calibrate when suite requests it)
oracle        -X-> boundary / drafter_mars
```

`drafter_mars` must **not** import private `_foo` symbols from `boundary`.
Everything it needs is a public export from `boundary` / `oracle`.

## API contracts (public)

### `evaluation.oracle`

| Symbol | Role |
| --- | --- |
| `OracleCandidate`, `DecodeStep`, `MethodResult` | types |
| `load_decode_traces`, `coerce_decode_steps` | I/O |
| `candidate_nonroot_path`, `candidate_matches_acceptance` | path helpers |
| `select_eagle3_only`, `select_perfect`, `select_sam_sequence_graft`, `select_budgeted`, `select_source_balanced` | selectors |
| `selected_mat` | MAT |
| `analyze_steps`, `render_table` | multi-method oracle table |

### `evaluation.boundary`

| Symbol | Role |
| --- | --- |
| `Prediction` | prediction record |
| `DEFAULT_NODE_BUDGET`, `MAX_TRIGGER_RATE`, `MIN_BOUNDARY_PRECISION` | constants |
| `eagle_candidates`, `sam_candidates` | accessors (was private) |
| `evaluate_predictions` | metrics |
| `choose_best`, `constrain_selection` | selection + Q2 guard (was private) |
| `run_threshold_sweep`, `threshold_grid` | sweep |
| `predict_low_margin_node` | baseline |
| `split_steps`, `require_calibration_schema` | split/schema |
| `calibrate` | default suite |

### `evaluation.drafter_mars`

| Symbol | Role |
| --- | --- |
| `ParentRecord` | parent record |
| `DRAFT_MARS_THETA_GRID`, `DRAFT_MARS_REACHABLE_QUANTILES` | grids |
| `predict_draft_mars_top_path`, `predict_draft_mars_reachable`, `predict_draft_delta_top_path` | predictors |
| `schema_report`, `raw_logit_diagnostics`, `require_drafter_mars_schema` | schema |
| `calibrate_drafter_mars` | suite |

## CLI entry points (post-change)

| Old | New |
| --- | --- |
| `python evaluation/oracle_fusion_analysis.py` | `python -m evaluation.oracle.cli` |
| `python evaluation/oracle_rejection_boundary.py` | `python -m evaluation.oracle.rejection_boundary` |
| `python evaluation/analyze_boundary_predictor.py` | `python -m evaluation.boundary.cli` |

Legacy top-level modules become **thin shims** that forward to the new modules so any stray remote command still works for one cycle; shims may be removed after scripts are updated.

## Delete set (evaluation + scripts + residuals)

### Evaluation sources — delete

Unrelated to Drafter-MARS Stage-1 offline path:

- `analyze_linear_alignment.py`, `analyze_vmiss.py`
- `equal.py`, `speed.py`
- `eval.py`, `eval_standard.py`, `eval_llama3.py`, `eval_vicuna.py`, `eval_naive_fusion.py`
- `inference_baseline.py`, `inference_eagle.py`, `inference_eagle2.py`, `inference_pld.py`, `inference_sam_only.py`, `inference_token_recycle.py`
- `medqa_prep.py`, `medquad_prep.py`
- `profile_entry.py`, `profile_sam_only.py`, `profile_samd.py`
- `oracle_high_precision_sam.py` (not used by Drafter-MARS baseline/smoke)
- `constrained_tasks/` (entire tree)
- `model/` evaluation mirrors (`evaluation/model/**`) if only used by deleted inference scripts — verify before delete; if `inference_samd` does not need them, delete
- all `*.bak` under evaluation/

### Keep under evaluation/

- new packages `oracle/`, `boundary/`, `drafter_mars/`
- `inference_samd.py` (smoke capture)
- `humaneval_prep.py`
- thin shims for old module paths (short-term)
- `data/` gitignored traces (not deleted by this change except optional local residuals outside git)

### Scripts — keep vs delete

**Keep:**

- `scripts/profile_boundary_predictor_humaneval.sh` (update module paths)
- `scripts/run_canonical_smoke_q0_20.sh` if it only orchestrates the above
- any script still required by smoke/canonical baseline docs after path update

**Delete (out of Drafter-MARS scope):**

- MT-Bench / MedQA / MedQuAD / overhead / naive-fusion multi-bench profile scripts
- eagle1/eagle2/pld/token_recycle/sam-only inference scripts
- boundary_graft online eval scripts
- vmiss / equal / speed scripts
- task-local `scripts/run_mtbench_current_compare_20260608.sh`

Exact list is enumerated in `implement.md` and applied only after a dry-run listing.

### Residuals

- tracked `*.bak`
- `.DS_Store`, `__pycache__`
- `.worktrees/2026-06-11-drafter-mars-sam-gate/` + `git worktree prune`

## Compatibility

- samd public behavior unchanged.
- Trace schema unchanged (`parent_top1_logit` / `parent_top2_logit` still optional).
- Analyzer metric keys unchanged so future result JSON stays comparable.
- Tests retarget imports to new packages; behavior assertions stay the same.

## Risks

| Risk | Mitigation |
| --- | --- |
| Circular import boundary ↔ drafter_mars | only lazy-import drafter_mars inside boundary CLI/calibrate when suite is requested |
| Broken remote smoke script | update `profile_boundary_predictor_humaneval.sh`; keep thin shims |
| Accidental delete of smoke dependency | keep `inference_samd.py` + samd; dry-run delete list first |
| depth_decoupled optional path | keep under `oracle/`; CLI still degrades gracefully if missing |

## Validation

1. `python3 -m py_compile` on all new packages + shims + inference_samd
2. Import smoke: `from evaluation.drafter_mars import calibrate_drafter_mars` etc.
3. `pytest` on boundary/oracle/naive tests when available; else py_compile + a tiny synthetic calibrate call
4. `bash -n scripts/profile_boundary_predictor_humaneval.sh`
5. Confirm deleted paths have no remaining in-repo references from kept scripts
