# MT-Bench Depth-Decoupled Oracle Result Summary

## Claim

Depth-decoupled fusion is not worth implementing for MT-Bench. The best split
improves over EAGLE-only by only `+0.61%`, below the 3% decision gate.

## Evidence

Dataset: MT-Bench q0-80 fusion profile trace.

Selection rule from the experiment spec:

```text
oracle gap > 3% -> consider implementation
oracle gap < 3% -> reject
```

Depth sweep:

| Method | MAT | Gap | Eagle Nodes | SAM Nodes | Prefix OK |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eagle_only` | 3.0798 | +0.00% | 59.00 | 0.00 | n/a |
| `depth_decoupled_d1` | 1.8493 | -39.95% | 6.94 | 3.08 | 62.04% |
| `depth_decoupled_d2` | 2.2335 | -27.48% | 23.55 | 2.99 | 43.88% |
| `depth_decoupled_d3` | 2.5058 | -18.64% | 36.92 | 2.91 | 30.93% |
| `depth_decoupled_d4` | 2.7015 | -12.28% | 45.61 | 2.83 | 22.76% |
| `depth_decoupled_d5` | 2.8474 | -7.55% | 51.08 | 2.75 | 17.04% |
| `depth_decoupled_d6` | 2.9553 | -4.04% | 54.63 | 2.67 | 13.19% |
| `depth_decoupled_d7` | 3.0356 | -1.44% | 57.09 | 2.59 | 10.32% |
| `depth_decoupled_d8` | 3.0986 | +0.61% | 59.00 | 2.51 | 7.83% |
| `depth_decoupled_d9` | 3.0798 | +0.00% | 59.00 | 2.42 | 0.00% |
| `depth_decoupled_d10` | 3.0798 | +0.00% | 59.00 | 2.34 | 0.00% |

Other oracle methods:

| Method | MAT | Gap |
| --- | ---: | ---: |
| `eagle3_only` | 3.0798 | +0.00% |
| `sam_sequence_graft` | 3.0883 | +0.27% |
| `perfect` | 3.1726 | +3.01% |
| `budgeted` | 3.1726 | +3.01% |
| `source_balanced` | 3.1726 | +3.01% |
| `depth_decoupled` | 3.0986 | +0.61% |

Compared with MedQA, MT-Bench had a better SAM match rate at the best split
(`7.83%` vs `3.01%`) and a higher perfect ceiling (`+3.01%` vs `+1.45%`), but
the actual depth-decoupled policy still missed the gate.

## Confounders

- This is a recorded-candidate oracle. It does not regenerate SAM from every
  EAGLE leaf.
- A positive `+0.61%` gap is still too small to cover expected implementation
  overhead and engineering risk.
- The perfect ceiling suggests there may be value in other SAM-use patterns,
  but not in tail-after-success depth decoupling.

## Decision

Reject depth-decoupled fusion for MT-Bench.

Depth-decoupled fusion is now closed for both evaluated datasets:

| Dataset | Decision |
| --- | --- |
| MedQA | Reject, `+0.00%` best gap |
| MT-Bench | Reject, `+0.61%` best gap |

## Code Retention

Keep `evaluation/oracle_depth_decoupled.py` as analysis infrastructure.

Do not implement `tree_fusion="eagle_leaf_sam_extend"` from this evidence.

## Avoid Repetition

Do not rerun depth-decoupled oracle on MT-Bench unless the trace generation or
SAM memory source changes materially.

## Next Step

Use the failure lesson to run Phase A alternative oracles:

- rejection-boundary SAM repair;
- high-precision SAM slice analysis.
