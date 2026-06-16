# Drafter-MARS SAM Gate Result Summary

Last updated: 2026-06-16

## Claim

The Drafter-MARS q0-20 smoke found a sparse boundary signal, but not enough
oracle MAT gain to justify a full HumanEval q0-164 run or online SAM grafting.

After syncing the auxiliary oracle scripts and rerunning the smoke, the result
package is complete enough to remove the "missing artifact" confounder. The
same-trace perfect and rejection-boundary ceilings are still low, so this smoke
is a negative gate result for the current q0-20 trace. Treat it as evidence that
this smoke split/config is not a useful launch point for full calibration, not
as a final rejection of every possible Drafter-MARS-style boundary predictor.

## Evidence

Source: user-provided run logs for:

```text
trace_json: evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.fusion_profile.json
summary_txt: evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.fusion_profile.txt
oracle_results: evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.oracle_results.json
rejection_boundary_results: evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.oracle_rejection_boundary.json
provenance_file: evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.provenance.txt
output_dir: evaluation/data/humaneval/drafter_mars_sam_gate/smoke
```

Run profile:

```text
decode steps: 1018
fusion_overhead_pct: 4.47%
node_budget: 60
predictor_suite: drafter_mars
train steps: 488
heldout steps: 530
```

Same-trace oracle summary:

```text
eagle3_only MAT: 6.7888
sam_sequence_graft MAT: 6.8075 (+0.27%)
perfect MAT: 7.0540 (+3.91%)
budgeted MAT: 7.0540 (+3.91%)
source_balanced MAT: 7.0540 (+3.91%)
depth_decoupled MAT: 6.8811 (+1.36%)
```

Rejection-boundary oracle summary:

```text
total decode steps: 1018
EAGLE rejection steps: 75 (7.4%)
SAM rescue success: 75
SAM rescue fail: 0
SAM rescue rate: 100.0%
baseline MAT (EAGLE-only): 6.7888
oracle MAT (with rescue): 7.0540
oracle gap: +3.91%
```

Selected gate:

```text
selected_method: draft_mars_reachable
selected_threshold: 0.96
selected_reachable_quantile: 0.85
selected_reachable_threshold: -0.002985060214996338
selected_by: constrained
selection_constraints_met: true
raw_logit_parent_count: 9251
draft_logit_ratio_parent_count: 9248
top1_nonpositive_rate: 0.03%
theta_thresholds: 8
delta_thresholds: 256
gate_profile: smoke
passing_methods: none
```

Held-out q10-20 metrics:

```text
heldout_trigger_rate: 13.21%
heldout_boundary_precision: 21.43%
heldout_boundary_recall: 28.85%
heldout_eagle_mat: 7.0811
heldout_perfect_mat: 7.3208
heldout_predicted_boundary_oracle_mat: 7.1396
heldout_mat_gain_per_trigger: 0.4429
heldout_mat_gain_per_sam_node: 0.031795
pass: false
```

Derived decision numbers:

```text
predicted MAT gain vs EAGLE: +0.83%
required smoke MAT gain: >= +2.00%
required predicted MAT: >= 7.2227
actual predicted MAT: 7.1396
perfect oracle gap vs EAGLE: +3.38%
recovered oracle ceiling: ~24.4%
```

The trigger rate and precision constraints passed, and raw-logit ratio stability
was acceptable. The printed gate failed because held-out predicted-boundary MAT
did not reach the `eagle_mat * 1.02` smoke gate.

The surprising part is the held-out same-trace perfect ceiling:

```text
heldout_perfect_mat - heldout_eagle_mat = 0.2397
perfect oracle gap vs EAGLE = +3.38%
```

Prior HumanEval oracle evidence on the q0-164 trace reported a `+9.65%`
perfect/rejection-boundary gap. A q0-20/q10-20 smoke can have sample variance,
but this low ceiling means the run may not contain the same SAM rescue evidence
that motivated the boundary-repair direction. The predictor cannot recover
oracle value that is absent or under-recorded in the trace.

The auxiliary oracle rerun confirms that SAM's value in this trace is still
concentrated exactly at EAGLE rejection boundaries: rejection-boundary oracle
gap equals the perfect oracle gap (`+3.91%`). The issue is not that the
boundary-repair hypothesis disappears on this trace; the issue is that the
recorded q0-20 ceiling is too small for a sparse predictor to pass the MAT gate.

Provenance from the rerun:

```text
git_commit: unknown
git_status_short_count: unknown
question_file_rows: 20
question_file_sha256: d49a763b8b6e702467421a38b412aa08858e26892427961e6d7bcdc70a36ac4a
boundary_analyzer_enabled: 0
```

## Local Code Audit Notes

The offline oracle is a recorded-candidate oracle, not a theoretical SAM oracle.
`evaluation/oracle_fusion_analysis.py` computes `perfect` by sorting the
candidate records present in the profile JSON under the same node budget.

In the naive fusion profile path, `samd/utils.py::gen_candidates()` skips SAM
candidate construction when the SAM match quality is below
`samd_len_threshold`. In those steps, the profile can record EAGLE-only
candidates. Therefore a low same-trace `perfect` gap can mean the q0-20 trace
did not record enough SAM rescue candidates, even if older HumanEval traces did.

This is why the missing `evaluation/oracle_rejection_boundary.py` run matters:
it would have reported whether the trace still contains the expected
EAGLE-rejection steps and SAM rescue successes.

## Auxiliary Analysis Rerun

The first pasted run was not fully clean:

```text
WARNING: depth_decoupled oracle skipped: No module named 'evaluation.oracle_depth_decoupled'
python: can't open file '.../evaluation/oracle_rejection_boundary.py': [Errno 2] No such file or directory
```

That failure came from missing untracked analyzer files in the experiment
worktree. After syncing `evaluation/oracle_rejection_boundary.py` and
`evaluation/oracle_depth_decoupled.py`, the rerun wrote the standalone
rejection-boundary oracle and included a depth-decoupled row in the main oracle
table.

Impact of the rerun:

- The missing-file confounder is resolved for this smoke.
- The same-trace rejection-boundary oracle equals the same-trace perfect oracle.
- The ceiling remains low (`+3.91%` overall, `+3.38%` held-out), so the failed
  Drafter-MARS MAT gate is not explained away by the missing scripts.

## Confounders

- The result is a q0-20 smoke, not a full q0-164 calibration.
- The held-out q10-20 split has only `530` decode steps,
  while an earlier q0-20 boundary-predictor smoke recorded `896` held-out decode
  steps, so generation length/config/provenance may differ.
- The same-trace perfect gap was only `+3.38%`, far below prior HumanEval
  oracle evidence (`+9.65%`).
- The metrics are offline oracle metrics from trace/profile output; they do not
  measure trace-off throughput.
- The rerun provenance does not expose git commit or dirty diff on the model
  host, so the exact checkout still cannot be tied to a commit hash from this
  artifact alone.

## Decision

- Do not run the full HumanEval q0-164 Drafter-MARS calibration from this q0-20
  smoke.
- Do not implement online `fusion_mode` behavior for Drafter-MARS SAM gating.
- If this direction is revisited, first explain why this q0-20 trace has much
  higher EAGLE-only MAT and lower same-trace oracle ceiling than the prior
  HumanEval q0-164 evidence.
- Keep trace enrichment and analyzer code as reusable offline tooling.

## Avoid Repetition

- Do not rerun the same q0-20 smoke as a model experiment unless a new
  hypothesis changes the trace split, candidate recording, or selection rule.
- Do not count sparse triggering alone as success; the printed MAT gain gate did
  not pass, even though the run still needs trace/provenance debugging.
- Do not report throughput speedup from this trace-on profile run.

## Next Step

Inspect `drafter_mars_smoke_q0_20.provenance.txt` and compare this q0-20 trace
against the prior q0-164 HumanEval oracle trace: question file hash, commit,
dirty diff, EAGLE MAT, decode-step count, and SAM candidate recording rate. Do
not proceed to full calibration until the low q0-20 ceiling is understood or a
new smoke split shows a ceiling closer to the validated HumanEval baseline.
