# Drafter-MARS Adaptive Gates — Final Report

Status: **closed by user on 2026-07-17**

## Decision

The research line is concluded. The best balanced HumanEval arm improved mean
accepted tokens by 6.28% but end-to-end throughput by only 2.0%. A simpler
theta-0.86, graft-8 arm reached the best measured throughput uplift at 2.61%.
The remaining oracle gap would require a learned, budget-aware selector and
substantial cross-domain GPU validation, which is not justified by the observed
system-level gain.

Phase B (learned multi-task gate) and further cross-domain batches are canceled,
not incomplete deliverables.

## Final numbers

HumanEval 0-164, greedy decoding, Llama-3.1-8B-Instruct + EAGLE3:

| Configuration | MAT | MAT vs baseline | Throughput vs baseline | Outcome |
| --- | ---: | ---: | ---: | --- |
| SAM[EAGLE3] baseline | 7.4849 | +0.00% | 1.0000x | Reference |
| theta 0.86, graft 8, K=1 | 7.7849 | +4.01% | 1.0261x | Best measured throughput point |
| repair 8, K=2, extension 16 | **7.9550** | **+6.28%** | **1.0199x** | Final balanced operating point |
| unbounded oracle | 8.4170 | +12.45% | about 0.66x | Selector ceiling only |

Throughput values must be compared within their recorded same-boot batches;
absolute tokens/s drifted across runs.

## Result notes

1. [Adaptive trio](docs/experiments/results/2026-07-15-adaptive-trio.md)
2. [Horizons, extension, subtree repair, and oracle ceiling](docs/experiments/results/2026-07-16-horizons-extension-oracle.md)
3. [Fixed verifier budget](docs/experiments/results/2026-07-16-fixed-verifier-budget.md)
4. [Preallocated EAGLE/SAM budget](docs/experiments/results/2026-07-17-preallocated-budget.md)

## What worked

- Drafter-side MARS ratio gating beat always-on grafting.
- Short chain repair plus medium leaf extension produced the best MAT/throughput
  tradeoff.
- K=2 multi-graft was useful; K=3 was marginal.
- The unbounded oracle confirmed that the candidate sources still contain more
  accepted-token potential.

## What did not work

- In-domain adaptive theta and hand-written depth/ratio budgets.
- Branching SAM subtrees: extra width did not beat a high-quality chain.
- Post-graft pruning to a fixed 60-node verifier tree.
- Preallocating EAGLE slots to SAM.
- Treating total node count as a sufficient proxy for verifier cost.

## Retained implementation

- Online Drafter-MARS gating, repair, extension, multi-graft, oracle mode, and
  optional fixed-budget pruning remain available as research/ablation code.
- All new behaviors are default-off; normal non-Drafter-MARS behavior is
  unchanged.
- Batch scripts retain environment-variable configuration, Feishu notification,
  and optional shutdown hooks without embedded credentials.

## Validation boundary

The recorded GPU experiments are the source of the performance claims. Local
closeout validation covers Python compilation, pure tests where dependencies
are available, shell syntax, documentation consistency, and secret scanning.
No new remote evaluation was requested or run during closeout.
