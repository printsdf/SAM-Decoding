# SAM + EAGLE3 Tree Fusion Result Summary

## Claim

- Tested `tree_fusion="sam_sequence_graft"` with `tree_method="eagle3"` on MT-Bench and MedQuAD.
- The run answers whether grafting the SAM matched sequence onto the EAGLE3 tree improves accepted length and wall-clock throughput over pure EAGLE3 and legacy SAMD+EAGLE3 routing.

## Evidence

- Baseline runs:
  - `baseline.jsonl` for each bench was used by `evaluation.speed` to compute tokens/s speedup.
  - Existing pure EAGLE3 and legacy SAMD+EAGLE3 results provided by the experimenter.
- New runs:
  - `evaluation/data/mt_bench/model_answer/samd_eagle3_fusion_p1.jsonl`
  - `evaluation/data/medquad/model_answer/samd_eagle3_fusion_p1.jsonl`
  - `evaluation/data/mt_bench/model_answer/samd_eagle3_fusion_p2.jsonl`
  - `evaluation/data/medquad/model_answer/samd_eagle3_fusion_p2.jsonl`
- Selection rule:
  - p1 trace OFF for wall-time / speedup.
  - p2 trace ON for tree step count and V_miss.
  - Same `MAX_NEW_TOKENS=512`, `MAX_CACHE_LEN=4096`, `samd_n_predicts=40`, `samd_len_threshold=5`, `samd_len_bias=5` for legacy SAMD+EAGLE3 and fusion.
  - Fusion command adds `--tree_fusion sam_sequence_graft`.
  - Evaluation uses greedy decoding (`temperature=0.0`, `num_choices=1`; evaluation code sets `torch.manual_seed(0)` for the single choice).
- Start commit:
  - `ecbcb8c1979efa7f1f258607c3cf7aad2af430f1` (`main` at experiment start, with local fusion changes in worktree).
- Model artifacts:
  - Base: `${MODEL_PATH}` from server `.env` (log shows Meta-Llama-3.1-8B-Instruct path configured on server).
  - Draft: `/root/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B` from server log.

## Metrics

| Bench | Group | mean_accept | tokens/s | speedup | tree_steps | V_miss rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mt_bench | eagle3 | 5.52 | 143.4 | 3.19x | 9704 | 21.5% |
| mt_bench | samd_eagle3 | 5.60 | 148.3 | 3.30x | 8544 | 18.7% |
| mt_bench | fusion | 5.854 | 154.7 | 3.44x | 9169 | 17.1% |
| medquad | eagle3 | 4.97 | 127.5 | 2.91x | 11874 | 28.8% |
| medquad | samd_eagle3 | 4.87 | 132.7 | 3.03x | 11476 | 28.0% |
| medquad | fusion | 5.012 | 138.4 | 3.16x | 11791 | 26.9% |

Fusion vs legacy SAMD+EAGLE3:

| Bench | mean_accept delta | tokens/s delta | speedup delta | V_miss delta |
| --- | ---: | ---: | ---: | ---: |
| mt_bench | +0.254 (+4.5%) | +6.4 (+4.3%) | +0.14x (+4.2%) | -1.6 pp |
| medquad | +0.142 (+2.9%) | +5.7 (+4.3%) | +0.13x (+4.4%) | -1.1 pp |

Fusion vs pure EAGLE3:

| Bench | mean_accept delta | tokens/s delta | speedup delta | V_miss delta |
| --- | ---: | ---: | ---: | ---: |
| mt_bench | +0.334 (+6.1%) | +11.3 (+7.9%) | +0.25x (+7.8%) | -4.4 pp |
| medquad | +0.042 (+0.8%) | +10.9 (+8.5%) | +0.25x (+8.6%) | -1.9 pp |

## Confounders

- Single run per group; variance across reruns is not measured.
- Pure EAGLE3 / legacy SAMD+EAGLE3 values were supplied from prior runs, not regenerated in the same shell command as fusion. They are comparable only if same dataset files, model artifacts, cache length, max tokens, and server load were used.
- p1 and p2 are intentionally separate because diagnosis tracing changes runtime. Do not mix p2 wall-time into speed claims.
- Greedy output equivalence against base model was not directly checked; the owner waived it as a blocking requirement because this experiment did not modify the target verifier / posterior acceptance rule. Residual risk remains in candidate buffer or cache-selection bugs.

## Decision

- Keep and extend.
- Evidence supports retaining Stage A `sam_sequence_graft`: both datasets improve throughput over legacy SAMD+EAGLE3 by about 4.3% and reduce V_miss.
- Before a stronger paper claim, rerun at least one repeated seed/run or regenerate all groups in the same job window to estimate noise.

## Code Retention

- Keep code changes.
- Experiment start commit: `ecbcb8c1979efa7f1f258607c3cf7aad2af430f1`.
- Revert condition if discarded: fusion must be reverted only if repeated apples-to-apples runs show no throughput or accepted-length benefit, or if a later correctness/equivalence check reveals output divergence caused by fusion buffers/cache selection.

## Avoid Repetition

- Do not repeat the p1/p2 split mistake: speed uses trace OFF; V_miss uses trace ON.
- Do not compare `evaluation.speed` rows from empty category slices (`nan`). For MT-Bench use `mt_bench` / `overall`; for MedQuAD use `medquad` / `overall`.

## Next Step

- Stage A is accepted with greedy-equivalence waiver; next experiment should be Stage B design for SAM multi-branch tree union + pruning, or a repeated apples-to-apples run if stronger performance evidence is needed.
