# Phase A4 results: preallocated EAGLE/SAM budget (2026-07-17)

HumanEval 0-164, same-boot baseline, theta 0.86. E53 means
`eagle3_total_token=53`; the remaining seven root-inclusive slots are reserved
for an r8 SAM continuation.

| arm | MAT | steps | tok/s | time_s | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline E60 | 7.4849 | 7890 | 214.67 | 275.1 | 1.0000 |
| append E60+r8/e16 | 7.9075 | 7462 | 218.34 | 270.3 | 1.0171 |
| prealloc E53+r8 | 7.6993 | 7646 | 214.34 | 274.7 | 0.9984 |
| prealloc E53+r8/e8 | 7.7765 | 7535 | 216.97 | 270.1 | 1.0107 |

## Readout

1. Prealloc E53+r8 is effectively baseline speed and loses most of the MAT
   gain. Reducing EAGLE before the gate removes useful candidates.
2. E53+r8/e8 reaches nearly the same wall time as append E60+r8/e16
   (270.1 vs 270.3 s), but MAT is 0.1310 lower (7.7765 vs 7.9075).
   It is therefore dominated by append-only repair.
3. A fixed total-node budget is not a sufficient abstraction: verifier cost is
   sensitive to tree shape and candidate coverage, not only node count.

## Decision

The Phase A budget-reallocation line is closed as a speedup route. Keep
append r8/e16 as the strongest current operating point/ablation. Further work
should either (a) use verifier-derived branch utility to choose exact EAGLE
subtrees (which is effectively a learned/oracle selection problem), (b) reduce
verifier kernel/tree-shape cost, or (c) reframe the contribution as higher MAT
at near-iso-latency rather than a substantial speedup.
