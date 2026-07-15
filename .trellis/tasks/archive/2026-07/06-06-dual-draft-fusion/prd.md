# Dual Draft Fusion PRD

Status: planning — awaiting approval before implementation
Last updated: 2026-06-19
Task: `dual-draft-fusion`

## Problem Statement

EAGLE3-only speculative decoding leaves accepted tokens on the table at the
points where the drafter is rejected by the target verifier. Same-trace oracle
analysis on canonical HumanEval (164 problems) shows a +2.72% MAT ceiling
recoverable by inserting SAM repair candidates exactly at EAGLE3's rejection
boundaries. The open question: can a **drafter-side** signal — computed before
target verification, with no second target forward pass — identify those
boundaries sparsely enough to recover a meaningful fraction of that ceiling,
without degenerating into always-on SAM insertion?

A prior attempt failed. The unconditioned "any EAGLE node has low margin"
trigger fired on 99.89% of steps (large trees almost always contain an
irrelevant low-margin branch), and the analyzer's unconstrained fallback then
reported the always-on threshold as if it had passed. Drafter-MARS is the
retry: it borrows MARS's top-2/top-1 raw-logit ratio signal, computed on the
drafter (not the target) and conditioned on reachable parents.

## Solution

An offline, trace-only calibration. Capture the EAGLE3 drafter's raw top-1/top-2
logits per parent, compute a MARS-style ratio gate, and measure — under a fixed
node budget on a same-trace replay — whether the gate recovers enough of the
rejection-boundary oracle ceiling at a low enough trigger rate to justify a full
run. No online fusion mode changes; no throughput claims.

This is Stage 1 (offline evidence) only. Stage 2 (whether offline MAT gain
translates to measured throughput) is a separate, currently blocked, online
experiment.

## User Stories

1. As a researcher, I want the EAGLE3 drafter to expose raw top-1/top-2 logits
   per parent, so that a MARS-style ratio can be computed without a second
   target forward pass.
2. As a researcher, I want the raw-logit capture to be profiler-gated and
   backward compatible, so that non-profiling inference paths are unchanged.
3. As a researcher, I want each captured tree node to carry its parent's top-1
   and top-2 raw logits, so that the analyzer can derive a parent-level ratio.
4. As a researcher, I want a `draft_mars_top_path` predictor that triggers only
   on greedy top-path parents, so that the gate cannot fire on irrelevant
   off-path branches (the prior failure mode).
5. As a researcher, I want a `draft_mars_reachable` ablation that triggers on
   high-reachability parents, so that non-greedy but likely branches are also
   testable.
6. As a researcher, I want a `draft_delta_top_path` fixed-logprob-delta control,
   so that the ratio signal can be distinguished from a plain margin.
7. As a researcher, I want the ratio computed as `z2 / (z1 + 1e-10)` on raw
   drafter logits (mirroring the MARS reference), so that the signal definition
   is faithful to its source.
8. As a researcher, I want `top1_nonpositive_rate` and `both_negative_rate`
   reported, so that ratio instability from negative drafter logits is visible
   rather than silent.
9. As a researcher, I want the selection rule to forbid unconstrained fallback,
   so that the always-on degeneration can never be reported as a pass.
10. As a researcher, I want selection constraints to equal the stage pass bar
    (smoke 20%/15%, full 15%/20%), so that a small smoke split is achievable and
    selection is always constrained.
11. As a researcher, I want recovered-oracle-ceiling measured same-trace and
    same-split, so that a q0-20 gain is never divided by a q0-164 ceiling.
12. As a researcher, I want a q0-20 smoke run with explicit pass/fail gates, so
    that I can decide whether to invest in a full q0-164 calibration.
13. As a researcher, I want the Drafter-MARS predictor to be a self-contained
    module, so that it is decoupled from the legacy calibration sweep.
14. As a researcher, I want the superseded legacy predictor methods removed
    (keeping `low_margin` only as the unconditioned-efficiency baseline), so
    that the analyzer carries one direction, not two.
15. As a researcher, I want the extended `gen_draft` return contract documented
    in the backend spec, so that future work knows the raw-logit path exists.
16. As a researcher, I want the smoke decision recorded as a result note with
    provenance (commit, split, thresholds, schema), so that a future reader can
    trust or refute the outcome.

## Implementation Decisions

**Method (mirrors MARS reference, `5SSjw/MARS`).** Per parent, drafter raw
top-1/top-2 logits `z1, z2`; trigger when `z2 / (z1 + 1e-10) > theta`. No
special-casing of negative logits (the MARS reference does none); negative
counts are diagnostics only. MARS itself is verifier-side on target logits with
relaxed acceptance; Drafter-MARS reuses only the ratio signal, on drafter
logits, as a SAM-insertion gate. It is not MARS. Threshold grid
`{0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98}` (matches MARS's
`relaxation_threshold` grid). Reachable variant sweeps quantiles
`{0.50, 0.70, 0.85, 0.95}` on `normalized_path_logprob`.

**Raw-logit capture contract.** `Eagle3Model.topK_genrate` gathers each parent's
top-2 raw logits from `last_headout` immediately after `torch.topk` (negligible
cost), aligned to the returned tree via the same `top_scores_index` used for
`draft_logprobs`. `Eagle3.gen_draft` exposes this through a new optional
`return_raw_logits` flag; the existing 2-tuple and 3-tuple return shapes are
unchanged, so non-profiling callers are untouched. This follows the established
pattern (phase 3.1 threaded `logprobs` through the same boundary).

**Trace enrichment.** `parse_eagle_tree` attaches `parent_top1_logit` /
`parent_top2_logit` to each candidate (the parent's z1, z2). The node's own
raw logit is not stored (MARS does not use it). Existing fields
(`local_logprob`, `sibling_margin`, `rank_among_siblings`,
`cumulative_path_logprob`) are preserved; old traces without raw logits still
load. `sibling_margin` already equals `z1 - z2` (logsumexp cancels) and serves
as a cross-check.

**OracleCandidate.** Adds optional `parent_top1_logit` / `parent_top2_logit`;
loader is backward compatible.

**Analyzer — modular Drafter-MARS suite.** A self-contained predictor module
implementing the three methods, parent extraction, threshold sweep, selection,
and diagnostics. Reuses the existing split / evaluate / same-budget-oracle
scaffolding. Selection rule: maximize train `predicted_boundary_oracle_mat`
subject to the stage's pass bar (smoke `trigger<=20%, precision>=15%`; full
`trigger<=15%, precision>=20%`). **If no threshold meets the bar,
`selected_by="none"`, no threshold is selected, and the heldout result is a hard
fail** — the unconstrained best is recorded as a diagnostic field only, never
used as the selected predictor. This removes the prior `unconstrained_diagnostic`
failure path.

**Legacy cleanup.** Superseded predictor methods (`high_margin_node`,
`risk_score_node`, `depth_only_low_confidence`, `depth_prior`) are removed from
the analyzer. `low_margin_node` is retained as the unconditioned-efficiency
baseline required by the "beats unconditioned trigger efficiency" gate.

**No online changes.** No `fusion_mode="drafter_mars"` or `boundary_graft`
online path is added or modified. All MAT numbers are same-trace oracle replays;
no throughput is reported.

**Scope / modules touched.** Drafter: `eagle3_model.py`, `eagle3.py` (capture +
return). Trace: `naive_fusion.py`, `utils.py`, `profiling/fusion_profiler.py`.
Analyzer: `oracle_fusion_analysis.py`, `analyze_boundary_predictor.py` + new
drafter_mars predictor module. Tests: extend `test_naive_fusion.py`,
`test_oracle_fusion_analysis.py`, `test_boundary_predictor_analysis.py`. Spec:
`tree-fusion.md` (document the `return_raw_logits` 4-tuple contract).

## Testing Decisions

Good tests assert external behavior on synthetic fixtures, not implementation
details, and never require a model. Highest feasible seams, preferring existing
files:

1. **Analyzer unit tests** (`test_boundary_predictor_analysis.py`, prior art:
   existing synthetic `DecodeStep` fixtures). Assert: ratio direction (larger
   theta → fewer triggers); ratio from raw logits not logprobs; top-path
   filtering ignores an off-path low-margin branch; `both_negative_rate` and
   `top1_nonpositive_rate` are reported; same-budget `predicted_MAT <= perfect_MAT`;
   and the Q2 guard — when no threshold meets the bar, `selected_by="none"` and
   no heldout pass is reported.
2. **Trace + loader unit tests** (`test_naive_fusion.py`, `test_oracle_fusion_analysis.py`;
   prior art: existing `local_logprob` tests). Assert: `parse_eagle_tree`
   populates `parent_top1/top2_logit` from a synthetic `eagle_tree` with raw
   logits; `OracleCandidate` preserves them; traces without raw logits still
   load.
3. **Drafter return-contract test** (extend the mock `gen_draft` already in
   `test_naive_fusion.py`). Assert: `return_raw_logits=True` yields the 4-tuple;
   default calls still return the 2- and 3-tuples unchanged.
4. **Remote smoke sweep** (server-only, acceptance not unit). q0-20 trace +
   three-method sweep via the existing profiling script; the analyzer's schema
   report must confirm raw drafter logits were captured before any gate is
   evaluated.

The real `topK_genrate` capture cannot be exercised locally without a model; it
is covered by seam 3 (contract) + seam 4 (real trace schema report).

## Smoke Pass Criteria (q0-20, held-out q10-20)

| Metric | Gate |
| --- | ---: |
| trigger rate | <= 20% |
| boundary precision | >= 15% |
| predicted MAT vs EAGLE | >= +1% |
| recovered oracle ceiling | >= 20% |
| beats unconditioned low_margin MAT/trigger | required |

`recovered_oracle_ceiling` uses the q0-20 same-trace rejection-boundary oracle
as denominator, never the q0-164 +2.72%. The predicted-MAT gate is the binding
constraint; recovered-ceiling is secondary. Full q0-164 pass (only after smoke
passes): trigger <= 15%, precision >= 20%, predicted MAT >= +2%, recovered >=
30%.

## Out of Scope

- No online `fusion_mode` change; no throughput measurement (Stage 2, blocked).
- No target logits in the predictor (drafter-side only, by design).
- No MT-Bench or MedQA (ceiling too low).
- No always-on / unconditioned trigger as a primary method.
- No rewriting of sound, reused scaffolding (trace fields, oracle accounting,
  split/eval/sweep framework).

## Further Notes

- **Baseline (canonical HumanEval 164, SHA256 `fc49f930...`):** EAGLE3-only MAT
  6.8180; `sam_sequence_graft` 6.8288 (+0.16%); rejection-boundary oracle
  7.0036 (+2.72%). The earlier +9.65% oracle is non-reproducible and is not a
  target.
- **MARS reference:** `5SSjw/MARS` (arXiv 2601.15498), verifier-side on target
  logits; Drafter-MARS reuses its ratio signal only. See `CONTEXT.md`.
- **Prior failure provenance:** unconditioned low-margin + unconstrained
  fallback → 99.89% trigger. Drafter-MARS conditions on reachable parents and
  forbids the fallback.
- Decisions Q1–Q5 and the modular-rewrite directive are captured in
  `CONTEXT.md` and this PRD; full step-by-step execution lives in the task's
  experiment plan.
