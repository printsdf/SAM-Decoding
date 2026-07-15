# Dual Draft Fusion — Experiment Result Report

**Task**: `06-06-dual-draft-fusion` (in_progress)
**Branch**: `feature/dual-draft-fusion` (HEAD `f9cb328`)
**Report date**: 2026-07-13
**Stage**: Stage 1 (offline evidence) — Drafter-MARS SAM gate

## 1. Objective

Test whether a **drafter-side** signal — computed on EAGLE3 drafter logits
*before* target verification, with no second target forward pass — can
sparsely identify EAGLE3's rejection boundaries well enough to recover a
useful fraction of the SAM repair ceiling, without degenerating into
always-on SAM insertion.

The method under test is **Drafter-MARS**: it borrows MARS's top-2/top-1
raw-logit ratio (`z2 / (z1 + 1e-10)`) but computes it on the **drafter**
(not the target) and uses it as a **SAM-insertion gate** (not as
verifier-side relaxed acceptance). It is not MARS.

Scope is offline, trace-only. No online `fusion_mode` change, no
throughput claim (Stage 2, blocked).

## 2. Baseline (canonical HumanEval, 164 problems, SHA256 `fc49f930...`)

Same-trace oracle replay, `node_budget=60`:

| Method | MAT | Nodes | MAT/node | Gap vs EAGLE |
| --- | ---: | ---: | ---: | ---: |
| **EAGLE3-only** | **6.8180** | 59.00 | 0.11556 | +0.00% |
| `sam_sequence_graft` | 6.8288 | 59.26 | 0.11524 | +0.16% |
| `depth_decoupled` | 6.8968 | 66.85 | 0.10317 | +1.16% |
| **perfect oracle** | **7.0036** | 59.26 | 0.11819 | **+2.72%** |

**Recovered-oracle-ceiling denominator = +2.72%** (rejection-boundary
oracle ceiling on the same split). The earlier +9.65% oracle
(2026-06-10, MAT 3.6689) is **not reproducible** — its question file and
config were lost; the MAT 3.67 vs 6.82 gap cannot be explained by
`max_new_tokens` alone. It is not a target.

## 3. Prior failure (the problem Drafter-MARS must not repeat)

Unconditioned "any EAGLE node has low margin" trigger + unconstrained
fallback, q0-20 smoke (held-out q10-20, `node_budget=60`):

| metric | value |
| --- | --- |
| trigger rate | **99.89%** |
| boundary precision | 25.47% |
| boundary recall | 79.44% |
| eagle MAT | 3.9833 |
| perfect MAT | 4.3449 |
| predicted boundary oracle MAT | 4.1406 |
| predicted vs EAGLE | +3.95% |
| recovered oracle ceiling | ~43.5% |
| `selected_by` | `unconstrained_diagnostic` |
| pass | **false** |

The trigger degenerated to always-on (large trees almost always contain
an irrelevant low-margin branch), and the analyzer reported the
always-on threshold as if it had passed via `unconstrained_diagnostic`.
**Same-split accounting held** (`3.9833 < 4.1406 < 4.3449`), but the
result was unusable as a sparse predictor. Full note:
`docs/archive/closed-experiments/results/2026-06-11-boundary-predictor-calibration.md`.

## 4. Implementation (complete, committed)

The Drafter-MARS path is fully implemented and committed on
`feature/dual-draft-fusion`. End-to-end chain:

1. **Drafter raw-logit capture** (`samd/tree_model/eagle3/eagle3.py`,
   `eagle3_model.py`): `gen_draft` / `topK_genrate` accept an optional
   `return_raw_logits` flag; gathers each parent's top-1/top-2 raw logits
   from `last_headout` right after `torch.topk`, aligned via the same
   `top_scores_index` used for `draft_logprobs`. Existing 2- / 3-tuple
   return shapes unchanged → non-profiling callers untouched.
2. **Capture activation** (`samd/utils.py`): passes
   `return_raw_logits=True` only on the profiling path; normal inference
   path unchanged.
3. **Trace enrichment** (`samd/fusion/naive_fusion.py`):
   `_normalize_raw_logits` + per-candidate `parent_top1_logit` /
   `parent_top2_logit`; preserves all existing fields
   (`local_logprob`, `sibling_margin`, `rank_among_siblings`,
   `cumulative_path_logprob`). Old traces without raw logits still load.
4. **Loader** (`evaluation/oracle_fusion_analysis.py`):
   `OracleCandidate` adds optional `parent_top1_logit` /
   `parent_top2_logit`; backward-compatible `from_dict`.
5. **Predictor module** (`evaluation/drafter_mars_predictor.py`, new):
   three methods + parent reconstruction + threshold/quantile sweep +
   schema diagnostics + selection. Reuses the split / evaluate /
   same-budget-oracle scaffolding in `analyze_boundary_predictor`; does
   not duplicate it.
6. **Analyzer CLI** (`evaluation/analyze_boundary_predictor.py`):
   `--predictor-suite {default,drafter_mars,all}`,
   `--theta-grid`, `--reachable-quantiles`,
   `--require-raw-draft-logits`, `--stage {smoke,full}`. Outputs
   `schema_report.json`, `selected_thresholds.json`,
   `heldout_metrics_*.json`, `train_sweep_*.json`,
   `boundary_predictor_calibration_summary.json`.
7. **Remote entry** (`scripts/profile_boundary_predictor_humaneval.sh`):
   smoke and full-HumanEval profiling + analyzer invocation.
8. **Tests** (`tests/test_boundary_predictor_analysis.py`,
   `test_naive_fusion.py`, `test_oracle_fusion_analysis.py`): cover
   ratio-from-raw-logits (not logprobs); larger-θ → fewer triggers
   (monotonic); top-path filtering ignores an off-path low-margin
   branch; `both_negative_rate` / `top1_nonpositive_rate` reported;
   same-budget `predicted_MAT <= perfect_MAT`; Q2 guard
   (`selected_by="none"`, `best_unconstrained_*` is diagnostic only);
   trace-loader preserves optional raw-logit fields; old traces still load.

### Three predictor methods (`evaluation/drafter_mars_predictor.py`)

| Method | Trigger rule | Role |
| --- | --- | --- |
| `draft_mars_top_path` | earliest greedy top-path parent with `z2/(z1+1e-10) > theta` | primary |
| `draft_mars_reachable` | earliest parent passing a `normalized_path_logprob = cumulative/depth` quantile **and** the ratio | ablation (1 extra HP) |
| `draft_delta_top_path` | earliest top-path parent with `logprob_top1 - logprob_top2 < tau` | fixed-delta control |

θ grid: `{0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98}` (mirrors
MARS `relaxation_threshold`). Reachable quantiles: `{0.50, 0.70, 0.85,
0.95}`. No special-casing of negative logits (MARS does none);
non-positive counts are diagnostics only.

### Selection + Q2 guard

Maximize train `predicted_boundary_oracle_mat` subject to the stage's
pass bar. **When no threshold meets the bar**, `selected_by="none"`,
no threshold is selected, and the heldout result is a **hard fail** —
the unconstrained best is recorded as `best_unconstrained_*` (diagnostic
only) and is never used as the selected predictor. This removes the
prior `unconstrained_diagnostic` failure path.

| bar | smoke (q0-20, held-out q10-20) | full (q0-164, held-out q82-164) |
| --- | ---: | ---: |
| trigger rate | ≤ 20% | ≤ 15% |
| boundary precision | ≥ 15% | ≥ 20% |
| predicted MAT vs EAGLE | ≥ +1% | ≥ +2% |
| recovered oracle ceiling | ≥ 20% | ≥ 30% |
| beats unconditioned low_margin MAT/trigger | required | required |

`recovered_oracle_ceiling` uses the **same-trace same-split**
rejection-boundary oracle as denominator, never the q0-164 +2.72%.

## 5. Smoke run (q0-20): status

The smoke trace **was produced on the remote GPU host** and is not in
the local checkout (`evaluation/data` is gitignored). Locally only the
provenance stub survives:

`evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.provenance.txt`

Provenance (remote):

```
cwd:           /root/SAM-Decoding/.worktrees/2026-06-11-drafter-mars-sam-gate
question_file: evaluation/data/humaneval/question.jsonl
question_file_sha256: d49a763b8b6e702467421a38b412aa08858e26892427961e6d7bcdc70a36ac4a
question_range: 0-20   train: 0-10   valid: 10-20
node_budget: 60   max_new_tokens: 512   max_cache_len: 4096   dtype: float16
```

> ⚠️ **Note**: this provenance records the question file SHA256
> `d49a763b...`, which is the **old q0-20 smoke** file (EAGLE MAT 6.79),
> **not** the canonical 164-question file `fc49f930...`. The smoke
> result therefore predates / is separate from the canonical baseline
> measurement. This is a provenance mismatch worth resolving on the
> next remote run (see §7).

The spec-required result document
(`docs/experiments/results/2026-06-11-drafter-mars-sam-gate.md`) **has
not been written**, and the analyzer output JSONs
(`heldout_metrics_q10_20.json`, `selected_thresholds.json`,
`schema_report.json`, `train_sweep_q0_10.json`,
`boundary_predictor_calibration_summary.json`) are not in the local
checkout. **No smoke pass/fail decision is recorded locally.**

## 6. Verification done in this session

- `python3 -m py_compile` on all six touch-points:
  `evaluation/drafter_mars_predictor.py`,
  `evaluation/analyze_boundary_predictor.py`,
  `evaluation/oracle_fusion_analysis.py`,
  `samd/fusion/naive_fusion.py`,
  `samd/tree_model/eagle3/eagle3.py`, `eagle3_model.py`,
  `samd/utils.py` → **OK**.
- `tests/test_boundary_predictor_analysis.py` inspected: asserts cover
  every spec seam (ratio from raw logits, θ monotonicity, off-path
  filtering, nonpositive/both-negative reporting, same-budget
  predicted≤perfect, Q2 guard, old-trace rejection).
- `pytest` is not installed in the local Python (`No module named
  pytest`); unit tests were **not** executed this session. Run them on
  the environment that has pytest before trusting any remote smoke
  number.

## 7. Open items / decisions to finish

1. **Re-run the smoke** on the remote host with the **canonical**
   question file (`fc49f930...`, or its q0-20 prefix). Record commit,
   dirty-diff summary, remote model paths, split, node budget, θ grid,
   reachable quantiles, schema report, selected method+threshold, train
   + heldout metrics, and the pass/fail decision into
   `docs/experiments/results/2026-06-11-drafter-mars-sam-gate.md`.
2. **Run the analyzer unit tests** where pytest is available.
3. **Smoke pass** (q10-20): trigger ≤ 20%, precision ≥ 15%, predicted
   MAT ≥ EAGLE×1.01, recovered ceiling ≥ 20%, beats low_margin
   MAT/trigger. Only then proceed to full q0-164 (bar: trigger ≤ 15%,
   precision ≥ 20%, predicted ≥ EAGLE×1.02, recovered ≥ 30%).
4. **Smoke fail path**: if all variants trigger above 20% or fail
   +1%/+2%, record a negative result and do **not** implement online
   Drafter-MARS SAM gating; keep the trace-loader + analyzer code as
   the offline harness.

## 8. Provenance

- git HEAD: `f9cb328` (branch `feature/dual-draft-fusion`), dirty on
  284 paths (mostly gitignored `.agents/skills/**` deletions + task
  docs; no source-code edits shown as unstaged in this session).
- Relevant committed work: `3e39a88` (drafter MARS calibration
  predictors), `10b8f77` (clarify Drafter-MARS experiment state),
`9166c0d` / `35154a2` (boundary + oracle fusion research utilities,
  experiment decisions), `547db75` / `f9cb328` (backend + medical prep
  evaluation references).
- MARS reference: `5SSjw/MARS` (arXiv 2601.15498), verifier-side on
  target logits; Drafter-MARS reuses its ratio signal only.

---

## 9. Modularization (same session, later)

Offline evaluation was reorganized into three packages (samd runtime
untouched):

```text
evaluation/oracle/        # types, load, select, analyze, rejection_boundary, depth_decoupled
evaluation/boundary/      # Prediction, metrics, selection/Q2 guard, low_margin, CLI
evaluation/drafter_mars/  # parents, predictors, schema, calibrate
```

Hard dependency rule: `drafter_mars -> boundary -> oracle` (no private
`_foo` imports across packages). Legacy top-level modules remain as
thin shims. Unrelated evaluation analyzers/inference scripts/scripts
and local residuals were deleted; smoke entry
`scripts/profile_boundary_predictor_humaneval.sh` was retargeted to
`python -m evaluation.*.cli`. Offline unit tests: **25 passed**.

**Bottom line.** The Drafter-MARS SAM-gate experiment is fully
implemented, modularized, and locally verified; the baseline (+2.72%
canonical ceiling) and the prior always-on failure are documented. The
smoke trace exists on the remote host but its held-out numbers have not
flowed back to the local checkout and no pass/fail decision is recorded.
The next concrete step is a remote smoke re-run on the canonical
question file, followed by writing the result note.
