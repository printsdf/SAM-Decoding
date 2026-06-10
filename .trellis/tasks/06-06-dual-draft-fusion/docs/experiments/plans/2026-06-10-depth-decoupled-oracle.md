# Depth-Decoupled Oracle Experiment Plan

**Goal:** Test whether depth-decoupled SAM tail extension has enough oracle
ceiling to justify implementation.

**Baseline:** `eagle_only` on the same fusion profile trace.

**Primary Metric:** Oracle MAT gap versus `eagle_only`; pass if MT-Bench gap is
greater than 3%.

**Budget:** One remote MT-Bench fusion-profile run over q0-80 plus offline
oracle analysis. Expected GPU runtime depends on host, but the runner uses
`max_new_tokens=512` and `max_cache_len=4096` by default.

---

## Files

| File | Purpose |
| --- | --- |
| `scripts/profile_fusion_mt_bench.sh` | Generates MT-Bench fusion profile trace and base oracle table |
| `evaluation/oracle_depth_decoupled.py` | Sweeps `D_split` for depth-decoupled oracle |
| `evaluation/oracle_fusion_analysis.py` | Shared profile loader and non-depth oracle baseline |
| `docs/experiments/results/` | Destination for the final result note |

## Remote Setup

Run on the remote GPU environment, not the local machine.

Before the no-tail gate, make sure these variables are unset unless the
experiment is explicitly tail-enabled:

```bash
unset EAGLE3_TAIL_PATH
unset EAGLE3_TAIL_TYPE
```

Set or verify model paths:

```bash
export MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct
export TREE_MODEL_PATH=/path/to/EAGLE3-LLaMA3.1-Instruct-8B
export MAX_CACHE_LEN=4096
```

## Step 1: Generate MT-Bench Fusion Profile

```bash
bash scripts/profile_fusion_mt_bench.sh
```

Expected artifacts:

```text
evaluation/data/mt_bench/profile_fusion_overhead/naive_fusion_mt_bench_q0_80.jsonl
evaluation/data/mt_bench/profile_fusion_overhead/naive_fusion_mt_bench_q0_80.fusion_profile.json
evaluation/data/mt_bench/profile_fusion_overhead/naive_fusion_mt_bench_q0_80.fusion_profile.txt
evaluation/data/mt_bench/profile_fusion_overhead/naive_fusion_mt_bench_q0_80.oracle_results.json
```

## Step 2: Run Depth-Decoupled Oracle

```bash
python evaluation/oracle_depth_decoupled.py \
  --trace-file evaluation/data/mt_bench/profile_fusion_overhead/naive_fusion_mt_bench_q0_80.fusion_profile.json \
  --output evaluation/data/mt_bench/profile_fusion_overhead/mt_bench_depth_decoupled.json \
  --plot-output evaluation/data/mt_bench/profile_fusion_overhead/mt_bench_oracle_gap.svg \
  --max-split 10
```

Inspect:

```bash
jq '.optimal_d_split, .oracle_results' \
  evaluation/data/mt_bench/profile_fusion_overhead/mt_bench_depth_decoupled.json
```

## Step 3: Record Result

Create a result note under:

```text
.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/
```

The note must include:

- claim
- baseline runs
- new run
- metrics and selection rule
- confounders
- decision: keep, rerun, debug, or reject
- exact next step

## Stop Conditions

- Stop and debug if profile JSON is missing candidates or acceptance paths.
- Stop and rerun if the trace accidentally enabled EAGLE3 tail sidecar for a
  no-tail decision.
- Reject depth-decoupled fusion for MT-Bench if the oracle gap is below 3%.
- Do not implement runtime fusion until the result note records a pass decision.

## Rollback Criteria

This plan is analysis-only. No code rollback is needed unless the profiling or
oracle scripts are modified during execution.
