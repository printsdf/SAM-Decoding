# Drafter-MARS SAM Gate Experiment Plan

**Goal:** Test whether MARS-style adaptive margins from EAGLE3 drafter raw
logits can gate SAM repair before target verification without the always-on
failure of unconditioned sibling-margin triggering.

**Baseline:** HumanEval q0-20 boundary-predictor smoke with same-trace
EAGLE-only held-out MAT `3.9833`, perfect held-out MAT `4.3449`, and the failed
unconditioned low-margin trigger MAT `4.1406` at `99.89%` trigger rate.

**Primary Metric:** Held-out q10-20 predicted-boundary oracle MAT after
selecting `theta` only on q0-10, subject to smoke trigger rate `<= 20%`.

**Budget:** One q0-20 smoke trace rerun if raw drafter logits are not already
available, plus CPU-only analyzer sweeps. Full q0-164 is allowed only after the
smoke pass gate.

---

## Scope

This plan is offline calibration only. It must not add online
`fusion_mode="drafter_mars"` behavior or make throughput claims from trace-on
profile output.

All model-dependent commands are intended to run inside the remote checkout. Do
not run model inference locally.

## Files

| File | Planned role |
| --- | --- |
| `samd/fusion/naive_fusion.py` | Preserve per-candidate raw EAGLE child logits and enough topology to reconstruct parent-level top-1/top-2 margins |
| `samd/utils.py` | Keep profiler-only logit work off the normal path |
| `samd/profiling/fusion_profiler.py` | Preserve optional trace fields without changing disabled-profiler overhead |
| `evaluation/oracle_fusion_analysis.py` | Load optional `local_logit` and parent-level derived fields without breaking old traces |
| `evaluation/analyze_boundary_predictor.py` | Add Drafter-MARS parent extraction and predictor suite |
| `scripts/profile_boundary_predictor_humaneval.sh` | Reuse or extend as the remote smoke entry point |
| `tests/test_naive_fusion.py` | Cover raw-logit trace fields when profiler is enabled |
| `tests/test_oracle_fusion_analysis.py` | Cover loader preservation of optional raw-logit fields |
| `tests/test_boundary_predictor_analysis.py` | Cover Drafter-MARS ratio direction, top-path filtering, and fallback behavior |

## Trace Contract

Prefer candidate-level raw-logit enrichment over a separate top-level parent
schema. Each EAGLE candidate should preserve:

```text
tree_index
parent_index
path_tree_indices
token_path
depth
token
local_logit
local_logprob
rank_among_siblings
cumulative_path_logprob
```

The analyzer derives parent records by grouping EAGLE candidates by
`parent_index`:

```text
parent_tree_index
parent_token_path
parent_depth
child_top1_token
child_top2_token
child_top1_logit
child_top2_logit
child_top1_logprob
child_top2_logprob
draft_logit_ratio = child_top2_logit / child_top1_logit
draft_adaptive_margin = child_top1_logit - child_top2_logit
is_top_path_parent
normalized_path_logprob
```

If `local_logit` is unavailable, the analyzer must mark Drafter-MARS ratio
variants unavailable and run only the fixed logprob-delta fallback.

## Step 1: Implement Raw-Logit Trace Enrichment

In `samd/fusion/naive_fusion.py`, add `local_logit` to EAGLE candidate metadata
when the profiler requests EAGLE logprobs/logits.

Rules:

- do not compute raw logits unless fusion profiling is enabled;
- preserve existing `local_logprob` and `sibling_margin` fields for backward
  compatibility;
- do not compute ratios inside trace capture; store raw values and let the
  analyzer derive parent statistics;
- set `local_logit = null` when raw logits are unavailable rather than
  fabricating a value from logprob.

Static checks:

```bash
python3 -m py_compile samd/fusion/naive_fusion.py samd/utils.py
python3 -m pytest tests/test_naive_fusion.py
```

Local note: if `pytest` is unavailable locally, record that and run the tests on
the remote environment.

## Step 2: Extend Oracle Loader

In `evaluation/oracle_fusion_analysis.py`, add optional `local_logit` to
`OracleCandidate`.

Rules:

- old traces without `local_logit` must still load;
- malformed raw-logit values should fail only the Drafter-MARS analyzer schema,
  not the generic oracle loader.

Checks:

```bash
python3 -m py_compile evaluation/oracle_fusion_analysis.py
python3 -m pytest tests/test_oracle_fusion_analysis.py
```

## Step 3: Add Drafter-MARS Analyzer Suite

In `evaluation/analyze_boundary_predictor.py`, add a separate predictor suite:

```text
draft_mars_top_path
draft_mars_reachable
draft_delta_top_path
```

Suggested CLI additions:

```text
--predictor-suite {default,drafter_mars,all}
--theta-grid 0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98
--reachable-quantiles 0.50,0.70,0.85,0.95
--require-raw-draft-logits
```

Predictor definitions:

```text
draft_mars_top_path:
  trigger earliest top-path parent with child_top2_logit / child_top1_logit > theta

draft_mars_reachable:
  trigger earliest parent passing reachability quantile and ratio > theta

draft_delta_top_path:
  trigger earliest top-path parent with child_top1_logprob - child_top2_logprob < tau
```

The selection rule remains train-only:

```text
maximize train predicted_boundary_oracle_mat
subject to trigger_rate <= 15% and boundary_precision >= 20%
```

Smoke pass is evaluated on held-out q10-20 with relaxed gates:

```text
trigger_rate <= 20%
boundary_precision >= 15%
predicted MAT >= eagle_mat * 1.02
recovered oracle ceiling >= 20%
```

Analyzer output must include:

```text
raw_logit_parent_count
top1_nonpositive_rate
selected_method
selected_theta
selected_reachable_quantile
heldout_eagle_mat
heldout_perfect_mat
heldout_predicted_boundary_oracle_mat
mat_gain_per_trigger
mat_gain_per_sam_node
```

Checks:

```bash
python3 -m py_compile evaluation/analyze_boundary_predictor.py
python3 -m pytest tests/test_boundary_predictor_analysis.py tests/test_oracle_fusion_analysis.py
```

## Step 4: Synthetic Local Sanity Tests

Before any model run, add synthetic tests proving:

- `theta` direction is correct: larger `theta` is stricter and triggers less;
- ratio is computed from raw logits, not logprobs;
- `child_top1_logit <= 0` marks ratio rows unstable;
- top-path filtering ignores an unrelated low-margin off-path branch;
- same-budget predicted MAT never exceeds perfect MAT.

Minimum local checks:

```bash
python3 -m py_compile \
  evaluation/analyze_boundary_predictor.py \
  evaluation/oracle_fusion_analysis.py \
  samd/fusion/naive_fusion.py \
  samd/utils.py

bash -n scripts/profile_boundary_predictor_humaneval.sh
git diff --check -- \
  evaluation/analyze_boundary_predictor.py \
  evaluation/oracle_fusion_analysis.py \
  samd/fusion/naive_fusion.py \
  samd/utils.py \
  scripts/profile_boundary_predictor_humaneval.sh \
  tests/test_boundary_predictor_analysis.py \
  tests/test_naive_fusion.py \
  tests/test_oracle_fusion_analysis.py
```

## Step 5: Remote q0-20 Smoke Trace

Run inside the remote checkout. Do not run locally.

```bash
PYTHON_BIN=python \
MODEL_PATH=/teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
TREE_MODEL_PATH=/teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
PROFILE_OUTPUT_DIR=evaluation/data/humaneval/drafter_mars_sam_gate \
MODEL_ID=drafter_mars_smoke_q0_20 \
QUESTION_BEGIN=0 \
QUESTION_END=20 \
MAX_NEW_TOKENS=512 \
MAX_CACHE_LEN=4096 \
RUN_BOUNDARY_ANALYZER=0 \
bash scripts/profile_boundary_predictor_humaneval.sh
```

Expected artifacts:

```text
evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.jsonl
evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.fusion_profile.json
evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.fusion_profile.txt
```

Stop if the trace schema report says no raw drafter logits were captured.

## Step 6: Remote q0-20 Analyzer Sweep

Run inside the remote checkout:

```bash
python evaluation/analyze_boundary_predictor.py \
  --trace-file evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.fusion_profile.json \
  --answer-file evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_smoke_q0_20.jsonl \
  --train-begin 0 \
  --train-end 10 \
  --valid-begin 10 \
  --valid-end 20 \
  --node-budget 60 \
  --predictor-suite drafter_mars \
  --theta-grid 0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98 \
  --reachable-quantiles 0.50,0.70,0.85,0.95 \
  --require-raw-draft-logits \
  --output-dir evaluation/data/humaneval/drafter_mars_sam_gate/smoke
```

Save:

```text
schema_report.json
selected_thresholds.json
heldout_metrics_q10_20.json
train_sweep_q0_10.json
boundary_predictor_calibration_summary.json
```

## Step 7: Smoke Decision

Pass if at least one Drafter-MARS variant satisfies on q10-20:

```text
trigger_rate <= 20%
boundary_precision >= 15%
predicted_boundary_oracle_mat >= heldout_eagle_mat * 1.02
recovered_oracle_ceiling >= 20%
mat_gain_per_trigger > unconditioned_low_margin_gain_per_trigger
```

Fail and stop if:

```text
trigger_rate > 20% for all ratio variants
or predicted MAT gain < +2% for all ratio variants
or top1_nonpositive_rate is large enough that ratios are unstable
```

If smoke fails, write:

```text
.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/2026-06-11-drafter-mars-sam-gate.md
```

## Step 8: Full HumanEval q0-164, Only After Smoke Pass

Run inside the remote checkout:

```bash
PYTHON_BIN=python \
MODEL_PATH=/teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
TREE_MODEL_PATH=/teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
PROFILE_OUTPUT_DIR=evaluation/data/humaneval/drafter_mars_sam_gate \
MODEL_ID=drafter_mars_humaneval_q0_164 \
QUESTION_BEGIN=0 \
QUESTION_END=164 \
MAX_NEW_TOKENS=512 \
MAX_CACHE_LEN=4096 \
RUN_BOUNDARY_ANALYZER=0 \
bash scripts/profile_boundary_predictor_humaneval.sh
```

Then analyze:

```bash
python evaluation/analyze_boundary_predictor.py \
  --trace-file evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_humaneval_q0_164.fusion_profile.json \
  --answer-file evaluation/data/humaneval/drafter_mars_sam_gate/drafter_mars_humaneval_q0_164.jsonl \
  --train-begin 0 \
  --train-end 82 \
  --valid-begin 82 \
  --valid-end 164 \
  --node-budget 60 \
  --predictor-suite drafter_mars \
  --theta-grid 0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98 \
  --reachable-quantiles 0.50,0.70,0.85,0.95 \
  --require-raw-draft-logits \
  --output-dir evaluation/data/humaneval/drafter_mars_sam_gate/full
```

Full pass requires:

```text
trigger_rate <= 15%
boundary_precision >= 20%
predicted_boundary_oracle_mat >= heldout_eagle_mat * 1.03
recovered_oracle_ceiling >= 30%
```

## Result Capture

Every result note must include:

```text
git commit
dirty diff summary
remote model paths
question range and split
node budget
theta grid
reachable quantiles
schema report
selected method and threshold
train metrics
heldout metrics
stop/pass decision
```

## Rollback Criteria

Discard the Drafter-MARS code path if:

- raw EAGLE logits cannot be captured without material profiler overhead;
- ratios are unstable because top-1 raw logits are often non-positive;
- all variants fail the q0-20 smoke trigger/MAT gates;
- the analyzer cannot keep same-budget oracle accounting coherent.

Keep trace-loader changes only if they remain backward compatible with existing
fusion-profile traces.
