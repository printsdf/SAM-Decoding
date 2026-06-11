# SAM + EAGLE3 Tree Union Pruning Result Summary

## Claim

- Tested `tree_fusion="sam_tree_union_prune"` with `tree_method="eagle3"` on MT-Bench and MedQuAD.
- The run answers whether Stage B's budget-pruned Dynamic SAM multi-branch tree union improves over Stage A's single `sam_sequence_graft` branch.
- With the first budget (`max_nodes=16`, `top_k=4`, `alpha=4.0`, `max_depth=6`), Stage B is worse than Stage A on both benches for throughput, mean accepted tokens, and V_miss. Treat this budget / first implementation as a negative result, not as an improvement.

## Evidence

- Baseline to beat:
  - Stage A `tree_fusion="sam_sequence_graft"`, recorded in `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/2026-06-01-sam-eagle3-tree-fusion.md`.
- New Stage B runs:
  - `evaluation/data/mt_bench/model_answer/samd_eagle3_union_p1.jsonl`
  - `evaluation/data/medquad/model_answer/samd_eagle3_union_p1.jsonl`
  - `evaluation/data/mt_bench/model_answer/samd_eagle3_union_p2.jsonl`
  - `evaluation/data/medquad/model_answer/samd_eagle3_union_p2.jsonl`
- Selection rule:
  - p1 trace OFF for wall-time / speedup.
  - p2 trace ON for tree step count and V_miss.
  - Same Stage A evaluation shape: `MAX_NEW_TOKENS=512`, `MAX_CACHE_LEN=4096`, `samd_n_predicts=40`, `samd_len_threshold=5`, `samd_len_bias=5`, greedy decoding.
  - Stage B command adds `--tree_fusion sam_tree_union_prune --sam_tree_max_nodes 16 --sam_tree_top_k 4 --sam_tree_alpha 4.0 --sam_tree_max_depth 6`.

## Metrics

| Bench | Group | mean_accept | tokens/s | speedup | tree_steps | V_miss rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mt_bench | Stage A sequence graft | 5.854 | 154.7 | 3.44x | 9169 | 17.1% |
| mt_bench | Stage B union-prune m16/k4 | 5.563 | 142.6 | 3.17x | 9646 | 19.0% |
| medquad | Stage A sequence graft | 5.012 | 138.4 | 3.16x | 11791 | 26.9% |
| medquad | Stage B union-prune m16/k4 | 4.980 | 137.3 | 3.14x | 11863 | 28.1% |

Stage B vs Stage A:

| Bench | mean_accept delta | tokens/s delta | speedup delta | tree_steps delta | V_miss delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| mt_bench | -0.291 (-5.0%) | -12.1 (-7.8%) | -0.27x | +477 (+5.2%) | +1.9 pp |
| medquad | -0.032 (-0.6%) | -1.0 (-0.7%) | -0.02x | +72 (+0.6%) | +1.2 pp |

## Interpretation

- The first Stage B budget does not satisfy the selection rule. It lowers throughput on both datasets and does not compensate with better mean accepted length or lower V_miss.
- MT-Bench is clearly worse: throughput drops about 7.8%, mean_accept drops about 5.0%, and V_miss increases by 1.9 pp.
- MedQuAD is closer on speed, but still directionally worse on every tracked metric: mean_accept, tokens/s, and V_miss.
- The likely mechanism is that the added SAM branches expand verifier work without improving the verifier-reachable next token distribution. In this form, multi-branch Dynamic SAM appears to add more low-value candidates than Stage A's single high-confidence sequence branch.

## Confounders

- Single run per group; run-to-run variance is not measured.
- Stage A and Stage B are comparable only if the same server model paths, dataset files, cache length, max tokens, and load conditions were used.
- The p1 / p2 split is intentional. Do not use p2 wall-time as a speed claim.
- `evaluation.speed` prints many `nan` category rows for empty slices. For MT-Bench use the `overall` / `mt_bench` row; for MedQuAD use the `medquad` / `overall` row.
- Fused node-count stdout was available, but the pasted summary did not include aggregate node-count values. The decision above uses the required core metrics: mean_accept, tokens/s, tree_steps, and V_miss.

## Decision

- Reject this first Stage B budget as an improvement over Stage A.
- Keep Stage A `sam_sequence_graft` as the current best fusion strategy.
- Do not claim multi-branch SAM tree union improves decoding from this evidence.

## Code Retention

- The implementation can stay as experiment scaffolding if we want further budget / pruning sweeps, but it should not be enabled by default.
- If the goal is a clean mainline optimized path, prefer disabling / removing Stage B unless a later redesign beats Stage A.

## Avoid Repetition

- Do not rerun the same `max_nodes=16`, `top_k=4`, `alpha=4.0`, `max_depth=6` budget expecting an improvement unless checking variance.
- If trying Stage B again, change the pruning objective, not only the tree size. Useful next hypotheses:
  - much smaller SAM budget, e.g. `max_nodes=4-8`, to test whether only the highest-frequency branch helps;
  - sibling budget only after matching an EAGLE3 prefix, to avoid adding unrelated root-level branches;
  - score candidates by verifier reachability diagnostics rather than raw Dynamic SAM `cnt_endpos` alone.

## Next Step

- For the current experiment thread, close Stage B as a negative result and proceed to acceptance / closeout if no further redesign is planned.
- If continuing research, write a new design for a narrower Stage B2 pruning rule instead of expanding the current union tree budget.
