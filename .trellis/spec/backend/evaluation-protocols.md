# Evaluation Protocols

> Benchmark and diagnosis rules for SAM-Decoding experiments.

## Diagnosis Trace

`SamdGenerationConfig.collect_diagnosis_trace` defaults to `False`. Keep it off
on normal inference paths. When enabled, each decode step records a small trace
with path type, accept length, rejected token fields, and verifier target
reachability.

The trace is intended for research analysis, not production inference.

## V_miss Definition

V_miss measures whether the verifier's desired next token is outside the EAGLE3
draft vocabulary during tree-step rejection. It is meaningful for EAGLE3 because
EAGLE3 may use a reduced draft vocab plus `d2t`/`t2d` mappings.

SAM-only has no draft vocabulary; report V_miss as N/A for that group.

## Phase Split Requirement

When a benchmark needs both wall-time speedup and V_miss, run two inference
phases on the same `question.jsonl`:

* phase1: trace OFF, used for wall-time speedup;
* phase2: trace ON, used for V_miss and tree-step diagnostics.

Use distinct model IDs and answer files, typically `_p1` and `_p2`, to prevent
overwrites. Do not calculate speedup from trace-ON outputs when the task also
compares trace-OFF groups.

Reason: trace adds a vocab-wide `argmax` and a `t2d` lookup per decode step.
The overhead is small, but it can bias low-margin speedup comparisons.

## Cross-Domain Benchmark Pattern

`scripts/run_bench_cross_domain_speedup.sh` is the reference batch shape:

1. fetch or prepare benchmark data once;
2. run phase1 trace-OFF inference for baseline, pure EAGLE3, SAM-only, and
   SAMD/EAGLE3 groups;
3. analyze speedup from phase1 outputs;
4. run phase2 trace-ON inference for EAGLE3 groups only;
5. analyze V_miss, reusing SAM-only phase1 as the N/A row;
6. send stage notifications and print final summaries.

## Dataset Requirements

* Use free-form prompts for vertical-domain V_miss studies when possible.
  Multiple-choice formats can understate terminology coverage problems because
  model outputs become formulaic.
* MedQuAD free-form medical questions showed about a 7 percentage point higher
  V_miss rate than MT-Bench in the recorded experiment.
* Do not evaluate answer correctness in V_miss/speedup scripts unless a task
  explicitly adds that metric. Current scripts measure acceleration and draft
  vocabulary coverage.

## Output and Analysis Contracts

* Answer files must preserve `choices[*].diagnosis_traces` when trace is on.
* `evaluation/analyze_vmiss.py` should tolerate malformed individual trace steps
  with warnings, but warn when an entire file lacks traces.
* Tables should report mean accept length, tree steps, and tree-step V_miss
  rate. Speedup tables should come from trace-OFF phase1.

## Scenario: Naive Fusion Profiling Gate

### 1. Scope / Trigger

Use this gate before deciding whether the naive logprob fusion path needs local
optimization or whether work can proceed to payoff-aware fusion. It applies to
`fusion_mode="naive"` with `tree_method="eagle3"`, `tree_fusion="none"`, and
`samd_len_threshold=5`.

### 2. Signatures

Run the profiler directly:

```bash
python scripts/profile_naive_fusion.py \
  --model-path "${MODEL_PATH}" \
  --tree-model-path "${TREE_MODEL_PATH}" \
  --question-begin 0 \
  --num-questions 10 \
  --max-cache-len 4096
```

Or use the cloudspace runner:

```bash
bash scripts/run_profile.sh
```

The runner reads `MODEL_PATH`, `TREE_MODEL_PATH`, optional `SAM_PATH`,
`BENCH_NAME`, `PROFILE_QUESTION_BEGIN`, `PROFILE_NUM_QUESTIONS`,
`MAX_NEW_TOKENS`, `MAX_CACHE_LEN`, `DTYPE`, `CUDA_VISIBLE_DEVICES`, and optional
`FEISHU_WEBHOOK_URL` from `.env` or the shell.

### 3. Contracts

The profiler must execute `evaluation.inference_samd` under `cProfile` with:

* `--tree_method eagle3`
* `--tree_fusion none`
* `--fusion_mode naive`
* `--samd_len_threshold 5`
* `--max_cache_len <= 4096` unless the caller explicitly allows a larger cache

It writes profile stats, answers, and a text report under
`evaluation/data/<bench>/profile_naive_fusion/` by default.

### 4. Validation & Error Matrix

* Missing `MODEL_PATH` or `TREE_MODEL_PATH` when running a profile ->
  argument error before model loading.
* `question_end <= question_begin` -> argument error.
* `max_cache_len > 4096` without an explicit large-cache override -> argument
  error to prevent cloudspace OOM.
* `--skip-run` with a missing stats file -> error instead of an empty report.

### 5. Good/Base/Bad Cases

* Good: profile 10 MT-Bench questions on the remote model host, inspect the
  report, and decide from the printed bottleneck percentages.
* Base: run `--dry-run` locally to inspect the exact `cProfile` command without
  importing model dependencies.
* Bad: profiling trace-ON diagnosis runs or omitting `--max_cache_len`, because
  this can bias runtime or allocate a huge KV cache.

### 6. Tests Required

* Compile `scripts/profile_naive_fusion.py`.
* Syntax-check `scripts/run_profile.sh` with `bash -n`.
* Run the profiler with `--dry-run` and verify the command includes
  `--fusion_mode naive`, `--tree_fusion none`, `--samd_len_threshold 5`, the
  requested question range, and `--max_cache_len`.
* Run the real profile only on the remote model environment.

### 7. Wrong vs Correct

Wrong:

```bash
python -m evaluation.inference_samd --fusion_mode naive
```

Correct:

```bash
python -m cProfile -o naive.stats -m evaluation.inference_samd \
  --tree_method eagle3 \
  --tree_fusion none \
  --fusion_mode naive \
  --samd_len_threshold 5 \
  --max_cache_len 4096
```

## Scenario: Fusion Profiler Oracle Traces

### 1. Scope / Trigger

This contract applies when `--profile-fusion` is enabled and the resulting
profile JSON is later consumed by `evaluation/oracle_fusion_analysis.py`.

### 2. Signatures

The evaluation CLI accepts:

```text
--profile-fusion
--fusion-profile-file <path>
--fusion-profile-summary-file <path>
```

The profile JSON step shape must include normal timing fields plus oracle fields:

```json
{
  "timings": {"draft_eagle": 0.0, "draft_sam": 0.0, "fusion_logic": 0.0, "verify": 0.0, "total": 0.0},
  "candidates": [{"source": "eagle", "token": 42, "depth": 1, "score": -0.2, "accepted": true}],
  "acceptance_path": [42],
  "stats": {"mat": 2, "eagle_nodes": 40, "sam_nodes": 20}
}
```

### 3. Contracts

`FusionProfiler` is opt-in and must not call `time.perf_counter()` on disabled
paths. Enabled profile runs record `draft_eagle`, `draft_sam`, `fusion_logic`,
`verify`, and `total` timings, plus cumulative `fusion_overhead_pct` and CUDA
memory deltas.

Oracle candidate records are non-root draft nodes. `acceptance_path` is also
non-root. Oracle MAT adds the always-emitted root/start token so it remains
comparable with evaluation `accept_lengths`.

### 4. Validation & Error Matrix

* `--profile-fusion` with multi-process evaluation -> fail before model load.
* profile step without `candidates` -> timing summary remains valid, but oracle
  analysis skips the step with a warning.
* malformed candidate source outside `eagle` or `sam` -> oracle analysis skips
  that step with a warning.
* disabled profiler -> no timing calls, no trace mutation, no JSON writes.

### 5. Good/Base/Bad Cases

* Good: run 10 MT-Bench samples remotely with `--profile-fusion`, then feed the
  JSON trace to `evaluation/oracle_fusion_analysis.py`.
* Base: locally compile the profiler/oracle modules and run synthetic JSON
  oracle checks without loading model dependencies.
* Bad: using trace-on profiling output as a throughput baseline for speedup
  claims; profile traces are for diagnosis, not final speedup tables.

### 6. Tests Required

* Compile `samd/profiling/fusion_profiler.py`,
  `evaluation/oracle_fusion_analysis.py`, and touched evaluation entry points.
* Syntax-check `scripts/profile_fusion_overhead.sh` with `bash -n`.
* Unit/smoke test that disabled `FusionProfiler` does not call
  `time.perf_counter()`.
* Unit/smoke test that oracle analysis loads profiler JSON steps and reports
  MAT with the root/start token included.

### 7. Wrong vs Correct

Wrong:

```python
candidate["accepted"] = candidate["token"] in new_tokens
```

Correct:

```python
candidate["accepted"] = candidate["token_path"][1:] == acceptance_path[:depth]
```

## Scenario: Boundary Predictor Calibration Traces

### 1. Scope / Trigger

This contract applies when `--profile-fusion` traces are used to calibrate an
offline rejection-boundary predictor before any online SAM grafting
implementation.

### 2. Signatures

Capture enriched traces with the task runner:

```bash
bash scripts/profile_boundary_predictor_humaneval.sh
```

Analyze an enriched trace with:

```bash
python evaluation/analyze_boundary_predictor.py \
  --trace-file <fusion_profile.json> \
  --answer-file <answer.jsonl> \
  --train-begin 0 \
  --train-end 82 \
  --valid-begin 82 \
  --valid-end 164 \
  --max-thresholds 256 \
  --node-budget 60 \
  --output-dir <output_dir>
```

### 3. Contracts

EAGLE candidate records must preserve existing oracle fields and add
`tree_index`, `parent_index`, `local_logprob`, `rank_among_siblings`,
`sibling_margin`, and `cumulative_path_logprob` when logprobs are available.
`tree_index` and `parent_index` refer to the original EAGLE tree, not the
post-fusion selected node order.

Profile step metadata must include a question split key such as
`metadata.question_index` so q0-82/q82-164 calibration can be reconstructed
from decode steps.

Boundary labels are top-level step fields:

```text
first_rejected_depth
first_rejected_tree_index
first_rejected_parent_index
first_rejected_parent_path
```

When the verifier accepts only the root/start token and no non-root token,
`first_rejected_depth` is `1`. When the verifier target token is absent from
the EAGLE children at the rejected depth, do not invent a rejected tree index;
record `first_rejected_tree_index = null` and identify the accepted parent
instead.

Margin semantics are fixed:

```text
margin = top1_logprob - top2_logprob
larger margin -> EAGLE is more certain
low-margin trigger -> sibling_margin < threshold
high-margin trigger -> sibling_margin > threshold
```

Thresholds applied to `exp(logprob)` values must be in `[0, 1]`.

The analyzer must bound threshold sweeps with `--max-thresholds` because
node-level margins/logprobs are dense floating-point values. Using every unique
score as a threshold can make a small smoke trace spend minutes or hours in CPU
sweeps.

Calibration MAT must use one shared node budget for EAGLE-only, perfect, and
predicted-boundary oracle selectors. Keep `--node-budget` aligned with
`evaluation/oracle_fusion_analysis.py --top-k` for the same trace. A predicted
boundary oracle MAT larger than the same-budget perfect MAT is an analyzer bug,
not a model result.

Threshold selection must first prefer rows satisfying the training trigger-rate
and precision constraints. If no row satisfies those constraints, the analyzer
may report the best unconstrained row only as a diagnostic and must mark it as
unconstrained in the output.

### 4. Validation & Error Matrix

* missing `question_index` on any step -> analyzer error.
* missing rejection-boundary labels -> analyzer error.
* missing node-level EAGLE fields -> analyzer error.
* old profile traces without enriched fields -> analyzer error, not silent
  depth-only fallback.
* probability threshold outside `[0, 1]` -> `ValueError`.
* dense threshold candidate set -> quantile/downsampled grid capped by
  `--max-thresholds`.
* predicted-boundary oracle MAT above same-budget perfect MAT -> analyzer bug;
  fix budget/selector accounting before interpreting results.

### 5. Good/Base/Bad Cases

* Good: enriched HumanEval trace records question ids, node confidence fields,
  and boundary labels; analyzer selects thresholds on q0-82 and reports held-out
  q82-164 metrics.
* Base: old oracle analysis still loads the same trace and computes MAT because
  original `candidates`, `acceptance_path`, and `stats` fields are preserved.
* Bad: using depth-only prediction as the main result when node/path labels are
  missing.

### 6. Tests Required

* Unit test that enriched `candidate_trace_records` exports EAGLE topology and
  confidence fields.
* Unit test that `FusionProfiler` step context records `question_index`.
* Unit test that oracle loader preserves enriched optional fields.
* Unit test that low-margin and high-margin predictors use opposite comparison
  directions.
* Unit test that node-level prediction can beat depth-only on a same-depth
  wrong-branch synthetic case.
* Unit test that dense threshold grids are capped by `--max-thresholds`.
* Compile `evaluation/analyze_boundary_predictor.py` and syntax-check
  `scripts/profile_boundary_predictor_humaneval.sh`.

### 7. Wrong vs Correct

Wrong:

```python
trigger = sibling_margin > margin_threshold
first_rejected_tree_index = first_child_under_parent
```

Correct:

```python
trigger = sibling_margin < margin_threshold
first_rejected_tree_index = None
first_rejected_parent_index = accepted_parent_tree_index
```

## Scenario: Depth-Decoupled Oracle Analysis

### 1. Scope / Trigger

Use this contract when measuring the oracle gap for leaf-only SAM tail extension
from fusion profile traces. The analysis is offline diagnosis, not a throughput
benchmark.

### 2. Signatures

Run the standalone analyzer:

```bash
python evaluation/oracle_depth_decoupled.py \
  --trace-file <fusion_profile.json> \
  --output <depth_oracle.json> \
  --plot-output <depth_oracle_gap.svg> \
  --max-split 10
```

The existing oracle analyzer also reports the optimal split selector:

```bash
python evaluation/oracle_fusion_analysis.py \
  --trace-file <fusion_profile.json> \
  --output-json <oracle_summary.json> \
  --depth-max-split 10
```

### 3. Contracts

The depth-decoupled oracle must count MAT as root/start token plus accepted
non-root draft tokens, matching `evaluation/oracle_fusion_analysis.py`.

`oracle_results` must include `eagle_only` and `depth_decoupled_d<N>` entries
for the swept split range. Each split entry reports MAT, gap versus Eagle-only,
EAGLE/SAM node budgets, prefix success rate, and SAM tail acceptance metrics.
The analyzer also writes an SVG oracle-gap plot and records its path in
top-level `plot_file`.

Top-level `depth_strata` is a split-level list for the optimal split, such as
`0-D` for the EAGLE prefix and `D+1-max` for the SAM tail. Per-depth diagnostics
belong under `depth_details`.

This is an oracle over recorded profile candidates. Existing traces do not
regenerate SAM from every accepted EAGLE leaf, so SAM tail credit is limited to
recorded SAM candidates whose path matches the verifier acceptance suffix.

### 4. Validation & Error Matrix

* missing or empty trace file -> fail with no decode steps found.
* malformed candidate source outside `eagle` or `sam` -> skip the step with a
  warning through the shared oracle loader.
* candidate with root-inclusive `token_path` -> compare `token_path[1:]` against
  `acceptance_path`.
* candidate with only root-inclusive parent `path` -> drop the root before
  reconstructing the non-root path.

### 5. Good/Base/Bad Cases

* Good: analyze remote MT-Bench or MedQA fusion profile traces and use the
  optimal split gap as a decision gate for full depth-decoupled fusion.
* Base: run synthetic JSON smoke tests locally without model dependencies.
* Bad: treating the recorded-candidate oracle as proof that SAM was actually
  regenerated from every EAGLE leaf.

### 6. Tests Required

* Compile `evaluation/oracle_depth_decoupled.py` and
  `evaluation/oracle_fusion_analysis.py`.
* Synthetic test that MAT includes the root/start token.
* Synthetic test that `depth_strata` is split-level while `depth_details` is
  per-depth.
* Regression test for root-inclusive `token_path` and fallback `path` handling.

### 7. Wrong vs Correct

Wrong:

```python
sam_extension = count_matching_sam_tokens_from_root()
```

Correct:

```python
sam_extension = count_matching_sam_tokens_after_accepted_eagle_prefix(d_split)
```
