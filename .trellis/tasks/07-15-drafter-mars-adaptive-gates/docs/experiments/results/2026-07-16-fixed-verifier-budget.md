# Phase A3 results: fixed 60-node verifier budget (2026-07-16)

HumanEval 0-164, same-boot baseline, theta 0.86. The budget is root-inclusive
and post-graft pruning protects all SAM nodes plus the EAGLE greedy path.

| arm | MAT | steps | tok/s | time_s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 7.4849 | 7890 | 215.65 | 273.8 | 1.0000 |
| append r8/e16 | 7.9075 | 7462 | 220.76 | 267.3 | 1.0237 |
| budget60 r8/e16 | 7.7564 | 7587 | 217.78 | 270.2 | 1.0099 |
| budget60 r8/K2/e16 | 7.7834 | 7630 | 218.55 | 271.7 | 1.0134 |

## Readout

1. The current fixed-budget policy is negative versus append-only. Budget60
   r8/e16 loses 0.1511 MAT and 1.38 percentage points of speedup.
2. Per-step wall time only falls from 35.82 ms (append) to 35.61 ms (budget60),
   while the step count rises by 1.68%. Keeping node count fixed therefore does
   not remove fusion/tree-shape overhead, and the lost EAGLE coverage dominates.
3. K2 does not repair the tradeoff: slightly higher MAT, but more steps and
   worse total time than budget60 K1.
4. Two implementation hypotheses explain the loss:
   - EAGLE3 selects its native tree by cumulative path logprob (`cu_scores`),
     whereas A3 pruning used mean path logprob.
   - Protect-all-SAM forces the full e16 extension tail to displace up to 16
     EAGLE nodes even when those tail nodes have lower marginal utility.

## Decision

- Do not continue K/horizon sweeps under the current post-prune policy.
- Next diagnostic arm should allocate the 60-node budget before verification:
  reduce `eagle3_total_token` and fill the reserved slots with short SAM repair,
  letting EAGLE's native cumulative-score selector choose the retained EAGLE
  subset. Candidate first arms: E53+r8 without extension and E53+r8/e8 with a
  budget-60 safety cap.
- If post-pruning is retained as an ablation, compare cumulative-path scoring
  and allow SAM tail pruning rather than protecting all SAM nodes.
