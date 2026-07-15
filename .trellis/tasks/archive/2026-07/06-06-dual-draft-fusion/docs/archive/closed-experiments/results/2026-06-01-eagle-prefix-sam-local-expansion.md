# EAGLE-Prefix SAM Local Expansion Result Summary

## Summary

Stage B2 `tree_fusion="eagle_prefix_sam_expand"` with budget `max_added_nodes=4`, `top_k=2`, `min_depth=1`, `max_depth=4` is a slight positive / near-neutral result compared with Stage A `sam_sequence_graft`.

It does not show the degradation observed in Stage B `sam_tree_union_prune`. Throughput is within the planned 1% tolerance on both benchmarks, mean accepted tokens improves slightly on both benchmarks, and MT-Bench V_miss improves marginally.

## Metrics

| Bench | Group | mean_accept | tokens/s | speedup | tree_steps | V_miss rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mt_bench | eagle3 | 5.52 | 143.4 | 3.19x | 9704 | 21.5% |
| mt_bench | samd_eagle3 | 5.60 | 148.3 | 3.30x | 8544 | 18.7% |
| mt_bench | Stage A `sam_sequence_graft` | 5.854 | 154.7 | 3.44x | 9169 | 17.1% |
| mt_bench | Stage B2 `eagle_prefix_sam_expand` m4/k2 | 5.856 | 154.3 | 3.43x | 9167 | 17.0% |
| medquad | eagle3 | 4.97 | 127.5 | 2.91x | 11874 | 28.8% |
| medquad | samd_eagle3 | 4.87 | 132.7 | 3.03x | 11476 | 28.0% |
| medquad | Stage A `sam_sequence_graft` | 5.012 | 138.4 | 3.16x | 11791 | 26.9% |
| medquad | Stage B2 `eagle_prefix_sam_expand` m4/k2 | 5.014 | 138.1 | 3.16x | 11786 | 26.9% |

## Delta vs Stage A

| Bench | mean_accept delta | tokens/s delta | V_miss delta |
| --- | ---: | ---: | ---: |
| mt_bench | +0.002 | -0.4 tokens/s (-0.25%) | -0.1 pp |
| medquad | +0.002 | -0.3 tokens/s (-0.24%) | 0.0 pp |

## Interpretation

- The Stage B2 local expansion preserves Stage A performance instead of regressing like full SAM tree union.
- The gains are very small, so this should not be claimed as a strong improvement from a single pass.
- The result is still useful: it validates the more conservative design principle of preserving Stage A candidate paths and only adding bounded local Dynamic-SAM siblings.

## Decision

Keep Stage B2 as a viable conservative variant, but do not replace Stage A as the default based on this run alone.

Recommended next step if continuing:

1. Run one repeat or a tiny budget sweep (`max_added_nodes=1/2/4`) to check whether the small MT-Bench V_miss gain is stable.
2. Keep Stage A `sam_sequence_graft` as the current strongest default until Stage B2 shows a repeatable gain beyond run noise.
3. Do not return to broad `sam_tree_union_prune` budget expansion unless the pruning objective changes substantially.
