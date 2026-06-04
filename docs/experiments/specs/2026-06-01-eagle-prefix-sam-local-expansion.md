# EAGLE-Prefix SAM Local Expansion Experiment Spec

## Goal

Test whether a small EAGLE-prefix-guided Dynamic SAM local expansion can improve over Stage A `sam_sequence_graft` without the cost and candidate-set drift observed in Stage B `sam_tree_union_prune`.

## Hypothesis

If we preserve the Stage A candidate tree and add only a few one-hop Dynamic SAM siblings at existing EAGLE/Stage-A internal prefix anchors, then V_miss or mean accepted tokens should improve without materially reducing tokens/s, because the added candidates are local alternatives at already-relevant prefixes and the Stage A leaf paths remain available.

## Baseline to Beat

Primary baseline:

- Stage A fusion: `tree_method="eagle3"`, `tree_fusion="sam_sequence_graft"`

Reuse the already completed baseline / Stage A runs; do not rerun them for Stage B2 unless a code-path regression is suspected.

| Bench | Group | mean_accept | tokens/s | speedup | tree_steps | V_miss rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mt_bench | eagle3 | 5.52 | 143.4 | 3.19x | 9704 | 21.5% |
| mt_bench | samd_eagle3 | 5.60 | 148.3 | 3.30x | 8544 | 18.7% |
| mt_bench | sam_sequence_graft | 5.854 | 154.7 | 3.44x | 9169 | 17.1% |
| medquad | eagle3 | 4.97 | 127.5 | 2.91x | 11874 | 28.8% |
| medquad | samd_eagle3 | 4.87 | 132.7 | 3.03x | 11476 | 28.0% |
| medquad | sam_sequence_graft | 5.012 | 138.4 | 3.16x | 11791 | 26.9% |

Negative reference:

- Stage B `sam_tree_union_prune` with `max_nodes=16/top_k=4/alpha=4.0/max_depth=6` lost on both benches; do not use it as the baseline to beat.

## Metrics and Selection Rule

Primary metrics:

- tokens/s and speedup from p1 trace-OFF runs.
- mean accepted tokens from p1 answer files.

Secondary diagnostics:

- tree_steps and V_miss from p2 trace-ON runs.
- local expansion stats: `anchor_count`, `added_nodes`, `retained_leaf_count`, `missing_leaf_count`, skipped duplicate siblings.

Selection rule:

- Keep Stage B2 only if tokens/s is within about 1% of Stage A on both benches, and at least one of mean_accept or V_miss improves on at least one bench.
- Reject or redesign if `sam_prefix_max_added_nodes=0` does not match Stage A behavior, if Stage A leaf retention fails, or if full p1/p2 loses on speed and diagnostics.

## Dataset / Split Assumptions

- MT-Bench: same `evaluation/data/mt_bench/question.jsonl` as Stage A.
- MedQuAD: same 200-question `evaluation/data/medquad/question.jsonl` as Stage A.
- Use `MAX_NEW_TOKENS=512`, `MAX_CACHE_LEN=4096`, greedy decoding (`temperature=0.0`, `num_choices=1`).
- Keep p1 / p2 split: p1 trace OFF for speed, p2 trace ON for V_miss and candidate diagnostics.

## Expected Code / Config Changes

- Add `tree_fusion="eagle_prefix_sam_expand"`.
- Add local expansion budget fields:
  - `sam_prefix_max_added_nodes` (default 4, allow 0 for Stage A equivalence diagnostic)
  - `sam_prefix_top_k` (default 2)
  - `sam_prefix_min_depth` (default 1)
  - `sam_prefix_max_depth` (default 4)
- Add `TreeSpec` helpers for leaf path extraction / retention check / adding child under existing internal nodes.
- Add Dynamic SAM prefix mapping helper to map existing Stage A tree prefixes to `(sam_index, sam_length)` anchors.
- Route Stage B2 in `DraftModel.lookup()` after Stage A baseline tree construction.
- Log expansion stats and leaf-retention stats.
- Do not change `Eagle3Model.topK_genrate()`, verifier forward, or `eval_posterior()`.

## Sanity Checks Before Full Runs

1. Unit-level leaf-retention check: Stage A leaf paths must all appear after local expansion.
2. Unit-level budget check: added nodes <= `sam_prefix_max_added_nodes`; per-anchor siblings <= `sam_prefix_top_k`; min/max depth respected.
3. Stage A equivalence diagnostic: `tree_fusion="eagle_prefix_sam_expand"` with `sam_prefix_max_added_nodes=0` should match Stage A candidate tokens/buffers on a small synthetic tree and a tiny benchmark subset.
4. Tiny MT-Bench subset with `max_added_nodes=4/top_k=2` to confirm stats are non-empty and no shape/device errors occur.

## Failure Modes / Invalidation Evidence

- `sam_prefix_max_added_nodes=0` differs from Stage A candidate tree.
- Local expansion makes any Stage A retrieve leaf path disappear.
- Added siblings mostly attach to irrelevant prefixes and V_miss does not improve.
- Added nodes lower throughput more than the selection rule allows.
- Dynamic SAM prefix mapping silently falls back to root-level expansion.

## First Minimal Experiment

Budget:

```text
sam_prefix_max_added_nodes=4
sam_prefix_top_k=2
sam_prefix_min_depth=1
sam_prefix_max_depth=4
```

Run order:

1. Tiny MT-Bench diagnostic with `sam_prefix_max_added_nodes=0`.
2. Tiny MT-Bench diagnostic with the first budget.
3. Full MT-Bench + MedQuAD p1 for Stage B2 only if diagnostics pass; compare against the reused table above.
4. Full MT-Bench + MedQuAD p2 for Stage B2 only if p1 is not clearly worse than Stage A; compare against the reused V_miss rows above.

## Server Run Commands

Set paths once before running the one-liners below:

```bash
export MODEL_PATH=/root/aicloud-data/Models/Llama-3.1-8B-Instruct TREE_MODEL_PATH=/root/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B MAX_NEW_TOKENS=512 MAX_CACHE_LEN=4096
```

Do not rerun `eagle3`, `samd_eagle3`, or `sam_sequence_graft`; use the completed table above as the baseline.

Tiny zero-budget diagnostic (MT-Bench first two questions; optional sanity check only):

```bash
python -m evaluation.inference_samd --template llama3 --model-type llama3 --model-path "$MODEL_PATH" --model-id samd_eagle3_prefix_zero_tiny --bench-name mt_bench --question-begin 0 --question-end 2 --answer-file evaluation/data/mt_bench/model_answer/samd_eagle3_prefix_zero_tiny.jsonl --max-new-tokens "$MAX_NEW_TOKENS" --max_cache_len "$MAX_CACHE_LEN" --tree_method eagle3 --tree_fusion eagle_prefix_sam_expand --tree_model_path "$TREE_MODEL_PATH" --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --sam_prefix_max_added_nodes 0
```

Tiny first-budget diagnostic:

```bash
python -m evaluation.inference_samd --template llama3 --model-type llama3 --model-path "$MODEL_PATH" --model-id samd_eagle3_prefix_m4k2_tiny --bench-name mt_bench --question-begin 0 --question-end 2 --answer-file evaluation/data/mt_bench/model_answer/samd_eagle3_prefix_m4k2_tiny.jsonl --max-new-tokens "$MAX_NEW_TOKENS" --max_cache_len "$MAX_CACHE_LEN" --tree_method eagle3 --tree_fusion eagle_prefix_sam_expand --tree_model_path "$TREE_MODEL_PATH" --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --sam_prefix_max_added_nodes 4 --sam_prefix_top_k 2 --sam_prefix_min_depth 1 --sam_prefix_max_depth 4
```

Full p1 speed run for Stage B2 only (replace `BENCH` with `mt_bench` or `medquad`; requires the existing base greedy answer file used in prior runs for `evaluation/speed.py`):

```bash
BENCH=mt_bench; python -m evaluation.inference_samd --template llama3 --model-type llama3 --model-path "$MODEL_PATH" --model-id samd_eagle3_prefix_m4k2 --bench-name "$BENCH" --answer-file evaluation/data/${BENCH}/model_answer/samd_eagle3_prefix_m4k2.jsonl --max-new-tokens "$MAX_NEW_TOKENS" --max_cache_len "$MAX_CACHE_LEN" --tree_method eagle3 --tree_fusion eagle_prefix_sam_expand --tree_model_path "$TREE_MODEL_PATH" --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --sam_prefix_max_added_nodes 4 --sam_prefix_top_k 2 --sam_prefix_min_depth 1 --sam_prefix_max_depth 4 && python evaluation/speed.py --file-path evaluation/data/${BENCH}/model_answer/samd_eagle3_prefix_m4k2.jsonl --base-path evaluation/data/${BENCH}/model_answer/baseline.jsonl --tokenizer-path "$MODEL_PATH"
```

Full p2 trace run for Stage B2 V_miss only (replace `BENCH` with `mt_bench` or `medquad`; compare against the reused V_miss rows above):

```bash
BENCH=mt_bench; python -m evaluation.inference_samd --template llama3 --model-type llama3 --model-path "$MODEL_PATH" --model-id samd_eagle3_prefix_m4k2_trace --bench-name "$BENCH" --answer-file evaluation/data/${BENCH}/model_answer/samd_eagle3_prefix_m4k2_trace.jsonl --max-new-tokens "$MAX_NEW_TOKENS" --max_cache_len "$MAX_CACHE_LEN" --tree_method eagle3 --tree_fusion eagle_prefix_sam_expand --tree_model_path "$TREE_MODEL_PATH" --samd_n_predicts 40 --samd_len_threshold 5 --samd_len_bias 5 --sam_prefix_max_added_nodes 4 --sam_prefix_top_k 2 --sam_prefix_min_depth 1 --sam_prefix_max_depth 4 --collect_diagnosis_trace
```

## Next Decision After First Run

- If zero-budget equivalence or leaf retention fails, fix implementation before any full run.
- If diagnostics pass but p1 speed drops materially, reduce `max_added_nodes` to 1-2 or stop.
- If p1 speed is close and p2 V_miss improves, run one repeated pass or a tiny budget sweep before making a stronger claim.
- If no metric improves, archive Stage B2 as another negative result and keep Stage A as the final strategy.
