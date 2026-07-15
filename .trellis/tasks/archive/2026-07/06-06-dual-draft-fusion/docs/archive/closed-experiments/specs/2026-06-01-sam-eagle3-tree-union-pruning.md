# SAM + EAGLE3 Tree Union Pruning Experiment Spec

## Goal

Test whether replacing Stage A's single SAM sequence graft with a pruned multi-branch SAM tree union improves speculative decoding on MT-Bench and MedQuAD.

## Hypothesis

If `tree_fusion="sam_tree_union_prune"` adds a budget-pruned Dynamic SAM multi-branch tree on top of the full EAGLE3 tree, then mean accepted tokens or V_miss should improve over Stage A `sam_sequence_graft`, because the verifier sees more online SAM continuations in the same decode step.

## Baseline to Beat

Primary baseline:

- Stage A fusion: `tree_method="eagle3"`, `tree_fusion="sam_sequence_graft"`

Reference baselines:

- Pure EAGLE3: `tree_method="eagle3"`, `tree_fusion="none"`, SAM threshold high enough to avoid SAM path.
- Legacy SAMD+EAGLE3: `tree_method="eagle3"`, `tree_fusion="none"`, normal SAM threshold / bias.

Last known Stage A results:

| Bench | Group | mean_accept | tokens/s | speedup | tree_steps | V_miss rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mt_bench | fusion | 5.854 | 154.7 | 3.44x | 9169 | 17.1% |
| medquad | fusion | 5.012 | 138.4 | 3.16x | 11791 | 26.9% |

## Metrics and Selection Rule

Primary metrics:

- tokens/s and speedup ratio from trace OFF p1 runs.
- mean accepted tokens per decode step from p1 answer files.

Secondary diagnostics:

- tree_steps and V_miss from trace ON p2 runs.
- fused tree node count, SAM added nodes, merged prefix nodes, and budget-hit rate.

Selection rule:

- Keep Stage B only if tokens/s is not worse than Stage A on both benches, and at least one of mean_accept or V_miss improves on at least one bench.
- Reject or redesign if tokens/s drops materially on both benches, even if mean_accept increases.

## Dataset / Split Assumptions

- MT-Bench: same `evaluation/data/mt_bench/question.jsonl` as Stage A run.
- MedQuAD: same 200-question `evaluation/data/medquad/question.jsonl` as Stage A run.
- Use `MAX_NEW_TOKENS=512`, `MAX_CACHE_LEN=4096`, greedy decoding (`temperature=0.0`, `num_choices=1`).
- Keep p1 / p2 split: p1 trace OFF for speed, p2 trace ON for V_miss and tree statistics.

## Expected Code / Config Changes

- Add `tree_fusion="sam_tree_union_prune"` and SAM tree budget fields to config / CLI / evaluation.
- Add Dynamic SAM tree draft support and budget fields; no `--sam_path` / StaticSAM artifact should be required for the first version.
- Port the useful `samd_sam_only` ideas selectively: `match_length * alpha` dynamic node count, online `cnt_endpos` ranking, `SearchItem` heap expansion, and per-depth `top_k` budget.
- Avoid a Dynamic-SAM `states_topk_next` cache in the first version; sort current outgoing edges by child `cnt_endpos` at query time to avoid cache invalidation.
- Add SAM tree draft helper with bounded expansion and budget pruning, returning `TreeSpec` instead of a separate tree buffer format.
- Extend `TreeSpec` with tree union by shared prefix.
- Route Stage B in `DraftModel.lookup()` without modifying verifier / `eval_posterior()`.
- Log budget parameters and fused tree statistics.

## Sanity Checks Before Full Runs

- Unit-level TreeSpec union checks: shared-prefix reuse, no duplicate same-parent token, path closure.
- SAM tree budget checks: max nodes, max depth, per-depth top-K, online `cnt_endpos` child ordering, no-`--sam_path` path.
- Compare heap-prior expansion against bounded BFS on a tiny synthetic SAM to make sure the chosen priority rule is observable and deterministic.
- Syntax/import checks in the target DL environment.
- Tiny question subset on MT-Bench and MedQuAD to confirm output files contain node-count diagnostics.

## Failure Modes / Invalidation Evidence

- Stage B creates much larger trees and lowers tokens/s on both benches.
- Dynamic SAM tree source silently falls back to Stage A sequence graft or unexpectedly requires a StaticSAM artifact.
- Prefix union creates duplicate same-parent token nodes or invalid retrieve paths.
- EAGLE3 pending hidden states are not consumed every decode step.
- V_miss improves but wall-clock speed drops enough to erase the practical benefit.

## First Minimal Experiment

Run Stage B with a conservative SAM budget, then compare against existing Stage A results:

- `sam_tree_max_nodes`: start small enough that fused node count is close to Stage A + limited SAM branches.
- `sam_tree_top_k`: start at 4 or 8.
- `sam_tree_alpha`: reuse SAM-only default idea, but record exact value.
- `sam_tree_max_depth`: cap to avoid very deep low-value branches.

Suggested first budget: `sam_tree_max_nodes=16`, `sam_tree_top_k=4`, `sam_tree_alpha=4.0`, `sam_tree_max_depth=6`. If tokens/s is flat or better, expand to `max_nodes=24` or `top_k=8` as the first budget sweep.

Run only MT-Bench first if implementation sanity is uncertain; otherwise run MT-Bench + MedQuAD p1, then p2 if p1 is not worse.

## Next Decision After First Run

- If Stage B beats Stage A on tokens/s or keeps tokens/s flat while reducing V_miss, tune budget grid.
- If Stage B improves mean_accept but slows down, reduce SAM node budget / depth and rerun.
- If Stage B loses on both speed and acceptance, stop and archive as a negative result before trying cross-source pruning.
