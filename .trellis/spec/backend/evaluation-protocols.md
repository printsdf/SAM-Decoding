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
