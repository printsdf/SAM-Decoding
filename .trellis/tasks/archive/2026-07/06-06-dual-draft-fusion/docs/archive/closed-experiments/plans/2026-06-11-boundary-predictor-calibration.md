# Boundary Predictor Calibration Experiment Plan

**Goal:** Test whether node-level EAGLE3 confidence/logprob features can predict
the true target-model rejection boundary well enough to justify online
Rejection-Boundary SAM Repair.

**Baseline:** HumanEval q0-164 same-trace EAGLE-only MAT `3.6689`, existing
`sam_sequence_graft` MAT `3.6812`, and perfect/rejection-boundary oracle MAT
`4.0231` from `results/2026-06-10-humaneval-oracle-results.md`.

**Primary Metric:** Held-out q82-164 predicted-boundary oracle MAT after
selecting thresholds only on q0-82; pass if it is at least
`eagle_only * 1.03` and recovers at least 30% of the oracle ceiling.

**Budget:** One 20-sample enriched-trace smoke run plus one full HumanEval
q0-164 enriched profile run; <= 5 GPU hours total on the remote model host.
Offline sweeps are CPU-only.

---

## Scope

This plan is offline calibration only. It must not change online generation
behavior, add `fusion_mode="boundary_graft"`, or report throughput claims from
profile traces.

All code, test, profiling, and analyzer commands in this plan are intended for
the remote checkout unless an individual command explicitly says otherwise.

The experiment answers one question:

```text
Do low-margin or risk-scored EAGLE nodes identify the first rejection boundary
better than random, depth-prior, depth-only, and high-margin controls?
```

If this fails on the held-out split, stop and write a negative result. Do not
implement online grafting.

## Files

| File | Planned execution role |
| --- | --- |
| `samd/fusion/naive_fusion.py` | Enrich debug `oracle_candidates` for EAGLE nodes with tree topology and confidence fields |
| `samd/fusion/types.py` | Only touch if `CandidateNode.metadata` is insufficient for enriched trace fields |
| `samd/samd_model.py` | Add verifier labels such as `first_rejected_depth` and optional `first_rejected_tree_index` to profile traces |
| `samd/profiling/fusion_profiler.py` | Only touch if per-run schema metadata or question context cannot pass through existing methods |
| `samd/utils.py` | Only touch if generation config needs a profile context such as `question_id` |
| `evaluation/eval_llama3.py` | Add enough profiler context to split trace steps by HumanEval question id |
| `evaluation/oracle_fusion_analysis.py` | Extend loader dataclasses to preserve optional node-level fields |
| `evaluation/oracle_rejection_boundary.py` | Reuse as a sanity oracle; do not replace unless labels disagree |
| `evaluation/analyze_boundary_predictor.py` | New offline analyzer for train/held-out threshold sweeps and oracle utility |
| `scripts/profile_boundary_predictor_humaneval.sh` | New remote entrypoint for enriched HumanEval trace capture |
| `tests/test_naive_fusion.py` | Unit coverage for enriched EAGLE candidate fields |
| `tests/test_fusion_profiler.py` | Unit coverage for trace schema and optional question context |
| `tests/test_oracle_fusion_analysis.py` | Loader compatibility for old and enriched traces |
| `tests/test_boundary_predictor_analysis.py` | New analyzer tests for margin direction, threshold scale, and node-vs-depth behavior |

## Trace Schema Contract

Each enriched decode step must keep the existing fields and add the following
where available.

EAGLE candidate fields:

```text
tree_index
parent_index
depth
token
path
token_path
local_logprob
rank_among_siblings
sibling_margin
cumulative_path_logprob
```

SAM evidence fields:

```text
sam_match_length
source_scores.sam
token_path
```

Verifier labels:

```text
acceptance_path
first_rejected_depth
first_rejected_tree_index
first_rejected_parent_index
```

`first_rejected_tree_index` is optional. If the target token is absent from the
EAGLE children at the rejection depth, label the boundary by
`first_rejected_parent_index` and path prefix instead of inventing a fake node.
The analyzer must report how many rejection steps have concrete node labels.

Margin semantics are fixed:

```text
margin = top1_logprob - top2_logprob
larger margin -> EAGLE is more certain
low-margin trigger -> sibling_margin < margin_threshold
```

If a score is `exp(logprob)`, every threshold for that score must be in `[0, 1]`.

## Artifacts

All generated experiment artifacts go under:

```text
evaluation/data/humaneval/boundary_predictor_calibration/
```

Required files:

```text
boundary_calibration_smoke_q0_20.jsonl
boundary_calibration_smoke_q0_20.fusion_profile.json
boundary_calibration_smoke_q0_20.fusion_profile.txt
boundary_calibration_humaneval_q0_164.jsonl
boundary_calibration_humaneval_q0_164.fusion_profile.json
boundary_calibration_humaneval_q0_164.fusion_profile.txt
schema_report.json
train_sweep_q0_82.json
heldout_metrics_q82_164.json
selected_thresholds.json
boundary_predictor_calibration_summary.json
```

Final result note:

```text
.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/2026-06-11-boundary-predictor-calibration.md
```

The result note must include command lines, git commit/hash or dirty diff
summary, seed/config, baseline reproduction, split metrics, confounders, and
decision.

## Step 1: Implement Trace Enrichment

Add enriched EAGLE candidate records in `samd/fusion/naive_fusion.py`.

Implementation rules:

- preserve the old trace fields so existing oracle scripts still parse;
- record original EAGLE `tree_index` and `parent_index`, not the post-fusion
  selected index;
- compute `rank_among_siblings` and `sibling_margin` among children with the
  same parent;
- set `sibling_margin` to `null` when there is only one sibling rather than
  fabricating a high-confidence value;
- compute `cumulative_path_logprob` as the sum of selected EAGLE node logprobs
  along the non-root path;
- add SAM fields without pretending SAM has EAGLE margins.

Add verifier labels in `samd/samd_model.py` by comparing EAGLE candidate
`token_path` values against `acceptance_path`. The first depth with no accepted
EAGLE prefix is `first_rejected_depth`.

Static checks:

```bash
python -m pytest tests/test_naive_fusion.py tests/test_fusion_profiler.py tests/test_oracle_fusion_analysis.py
python -m py_compile samd/fusion/naive_fusion.py samd/samd_model.py samd/profiling/fusion_profiler.py evaluation/oracle_fusion_analysis.py
```

Stop if enriched traces cannot identify question ids or reconstruct the
q0-82/q82-164 split.

## Step 2: Implement Offline Analyzer

Create `evaluation/analyze_boundary_predictor.py`.

Required CLI:

```bash
python evaluation/analyze_boundary_predictor.py \
  --trace-file evaluation/data/humaneval/boundary_predictor_calibration/boundary_calibration_humaneval_q0_164.fusion_profile.json \
  --answer-file evaluation/data/humaneval/boundary_predictor_calibration/boundary_calibration_humaneval_q0_164.jsonl \
  --train-begin 0 \
  --train-end 82 \
  --valid-begin 82 \
  --valid-end 164 \
  --output-dir evaluation/data/humaneval/boundary_predictor_calibration
```

The analyzer must evaluate:

| Predictor | Rule |
| --- | --- |
| random node | sample triggered EAGLE nodes at matched trigger rate; report mean over fixed seeds |
| depth prior | choose the most common training rejection depth |
| depth-only low confidence | choose depth from aggregate confidence, then first path at that depth |
| high-margin node | trigger when `sibling_margin > threshold` |
| low-margin node | trigger when `sibling_margin < threshold` |
| risk-score node | combine `-sibling_margin`, `-local_logprob`, normalized path logprob, and depth penalty |

Threshold selection must happen only on q0-82. Select the threshold with the
best train predicted-boundary oracle MAT subject to:

```text
trigger_rate <= 15%
boundary_precision >= 20%
```

Tie break by higher boundary recall, then lower trigger rate.

Analyzer tests:

```bash
python -m pytest tests/test_boundary_predictor_analysis.py tests/test_oracle_fusion_analysis.py
python -m py_compile evaluation/analyze_boundary_predictor.py
```

The tests must prove:

- low margin means `sibling_margin < threshold`;
- high-margin trigger is a separate control, not the default margin direction;
- probability thresholds used with `exp(logprob)` reject values outside `[0, 1]`;
- node-level prediction can beat depth-only on a synthetic same-depth wrong-path
  case;
- old profile traces without enriched fields fail with a clear schema error for
  calibration, not with silent depth-only fallback.

## Step 3: Remote Smoke Run

Run on the remote GPU host, not on the local machine.

```bash
ssh -i /Users/printsdf/.ssh/id_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  s_01ktapxr3q3898aj388psbxye3@ssh.lightning.ai \
  'cd /teamspace/studios/this_studio/SAM-Decoding && \
   PYTHON_BIN=python \
   MODEL_PATH=/teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
   TREE_MODEL_PATH=/teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
   PROFILE_OUTPUT_DIR=evaluation/data/humaneval/boundary_predictor_calibration \
   MODEL_ID=boundary_calibration_smoke_q0_20 \
   QUESTION_BEGIN=0 \
   QUESTION_END=20 \
   MAX_NEW_TOKENS=512 \
   MAX_CACHE_LEN=4096 \
   bash scripts/profile_boundary_predictor_humaneval.sh'
```

Then run the analyzer on the smoke trace:

```bash
ssh -i /Users/printsdf/.ssh/id_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  s_01ktapxr3q3898aj388psbxye3@ssh.lightning.ai \
  'cd /teamspace/studios/this_studio/SAM-Decoding && \
   python evaluation/analyze_boundary_predictor.py \
     --trace-file evaluation/data/humaneval/boundary_predictor_calibration/boundary_calibration_smoke_q0_20.fusion_profile.json \
     --answer-file evaluation/data/humaneval/boundary_predictor_calibration/boundary_calibration_smoke_q0_20.jsonl \
     --train-begin 0 \
     --train-end 10 \
     --valid-begin 10 \
     --valid-end 20 \
     --output-dir evaluation/data/humaneval/boundary_predictor_calibration/smoke'
```

Smoke pass conditions:

- profile JSON has nonempty `candidates` and `acceptance_path`;
- `question_id` or equivalent split metadata is present;
- at least one EAGLE rejection step is found;
- schema report shows required EAGLE fields present for multi-sibling nodes;
- old oracle analysis still runs on the same trace.

## Step 4: Full HumanEval Run

Only run after smoke passes.

```bash
ssh -i /Users/printsdf/.ssh/id_ed25519 \
  -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  s_01ktapxr3q3898aj388psbxye3@ssh.lightning.ai \
  'cd /teamspace/studios/this_studio/SAM-Decoding && \
   PYTHON_BIN=python \
   MODEL_PATH=/teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
   TREE_MODEL_PATH=/teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
   PROFILE_OUTPUT_DIR=evaluation/data/humaneval/boundary_predictor_calibration \
   MODEL_ID=boundary_calibration_humaneval_q0_164 \
   QUESTION_BEGIN=0 \
   QUESTION_END=164 \
   MAX_NEW_TOKENS=512 \
   MAX_CACHE_LEN=4096 \
   bash scripts/profile_boundary_predictor_humaneval.sh'
```

Inside the same remote checkout, reproduce oracle baselines:

```bash
python evaluation/oracle_fusion_analysis.py \
  --trace-file evaluation/data/humaneval/boundary_predictor_calibration/boundary_calibration_humaneval_q0_164.fusion_profile.json \
  --output-json evaluation/data/humaneval/boundary_predictor_calibration/oracle_results.json

python evaluation/oracle_rejection_boundary.py \
  --trace-file evaluation/data/humaneval/boundary_predictor_calibration/boundary_calibration_humaneval_q0_164.fusion_profile.json \
  --output evaluation/data/humaneval/boundary_predictor_calibration/oracle_rejection_boundary.json
```

Sanity gates:

```text
EAGLE-only MAT ~= 3.6689 +/- 0.02 on q0-164
rejection-boundary oracle gap ~= +9.65% within reasonable trace variance
EAGLE rejection step count close to 432 / 6932
```

Stop and debug if baseline reproduction fails.

## Step 5: Calibration Sweep

Inside the same remote checkout, run the analyzer on the full trace:

```bash
python evaluation/analyze_boundary_predictor.py \
  --trace-file evaluation/data/humaneval/boundary_predictor_calibration/boundary_calibration_humaneval_q0_164.fusion_profile.json \
  --answer-file evaluation/data/humaneval/boundary_predictor_calibration/boundary_calibration_humaneval_q0_164.jsonl \
  --train-begin 0 \
  --train-end 82 \
  --valid-begin 82 \
  --valid-end 164 \
  --output-dir evaluation/data/humaneval/boundary_predictor_calibration
```

Report held-out q82-164 metrics:

| Metric | Pass criterion |
| --- | --- |
| trigger rate | `<= 15%` |
| boundary precision | `>= 20%` |
| boundary recall | `>= 35%` |
| precision lift | `>= 3x` rejection-step base rate |
| median depth error | `<= 1` |
| path-prefix hit rate | `>= 50%` on true positives |
| predicted-boundary oracle MAT | `>= EAGLE-only * 1.03` |
| recovered oracle ceiling | `>= 30%` |
| low-margin vs high-margin | low-margin beats high-margin |
| node-level vs depth-only | node-level beats depth-only |

## Step 6: Result Note and Decision

Write:

```text
.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/2026-06-11-boundary-predictor-calibration.md
```

Decision options:

| Decision | Evidence |
| --- | --- |
| Proceed to online graft planning | All held-out signal and oracle utility gates pass |
| Redesign predictor | Low-margin fails but high-margin or risk-score clearly wins |
| Keep analyzer, reject online graft | Signal or oracle utility gates fail |
| Debug trace schema | Baseline reproduction or labelability fails |

If proceeding, freeze the selected threshold/risk score in the result note. Do
not retune it during the first online graft experiment.

## Rollback and Stop Conditions

Stop immediately if:

- enriched traces lack question split metadata;
- `first_rejected_depth` disagrees with the existing rejection-boundary oracle;
- `exp(logprob)` thresholds outside `[0, 1]` appear in config or output;
- low-margin and high-margin are accidentally implemented with the same
  comparison direction;
- smoke trace is empty, lacks candidates, or cannot run old oracle scripts;
- full-trace EAGLE-only MAT differs from `3.6689` by more than `0.02`;
- held-out calibration fails the pass criteria above.

Rollback criteria:

- If trace-enrichment code breaks existing naive fusion tests or old oracle
  analysis, revert code changes before rerunning the experiment.
- If calibration fails but infrastructure is correct, keep the analyzer and
  result note as a negative result, but do not proceed to online grafting.

## Handoff

Execution should use `superpowers:experiment-execution` next. The first
implementation batch should stop after Step 2 tests pass; run the remote smoke
only after the trace schema is locally testable.
