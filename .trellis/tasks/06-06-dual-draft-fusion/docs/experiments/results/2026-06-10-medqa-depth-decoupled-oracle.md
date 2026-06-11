# MedQA Depth-Decoupled Oracle Result Summary

## Claim

Depth-decoupled fusion has no useful oracle ceiling on the MedQA q0-80 trace.
The best split matches EAGLE-only at `+0.00%` oracle gap, so this direction
should be rejected for MedQA.

## Evidence

Baseline: `eagle_only` on the same MedQA fusion profile trace.

Selection rule from the experiment spec: continue only if oracle MAT gap exceeds
3%.

Depth-decoupled sweep:

| Method | MAT | Gap | Eagle Nodes | SAM Nodes | Prefix OK |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eagle_only` | 2.6771 | +0.00% | 59.00 | 0.00 | n/a |
| `depth_decoupled_d1` | 1.6946 | -36.70% | 7.28 | 2.39 | 61.38% |
| `depth_decoupled_d2` | 2.0651 | -22.86% | 25.28 | 2.33 | 40.27% |
| `depth_decoupled_d3` | 2.2977 | -14.17% | 39.55 | 2.26 | 25.21% |
| `depth_decoupled_d4` | 2.4495 | -8.50% | 48.37 | 2.20 | 16.34% |
| `depth_decoupled_d5` | 2.5453 | -4.92% | 53.40 | 2.14 | 10.39% |
| `depth_decoupled_d6` | 2.6074 | -2.60% | 56.22 | 2.07 | 6.64% |
| `depth_decoupled_d7` | 2.6488 | -1.06% | 57.92 | 2.01 | 4.47% |
| `depth_decoupled_d8` | 2.6772 | +0.00% | 59.00 | 1.95 | 3.01% |
| `depth_decoupled_d9` | 2.6771 | +0.00% | 59.00 | 1.88 | 0.00% |
| `depth_decoupled_d10` | 2.6771 | +0.00% | 59.00 | 1.82 | 0.00% |

Other oracle baselines:

| Method | MAT | Gap |
| --- | ---: | ---: |
| `eagle3_only` | 2.6771 | +0.00% |
| `sam_sequence_graft` | 2.6855 | +0.31% |
| `perfect` | 2.7158 | +1.45% |
| `budgeted` | 2.7158 | +1.45% |
| `source_balanced` | 2.7158 | +1.45% |
| `depth_decoupled` | 2.6772 | +0.00% |

## Confounders

- This is an oracle over recorded profile candidates. It does not regenerate SAM
  from every accepted EAGLE leaf.
- The result applies to MedQA and should not be generalized to MT-Bench without
  a separate same-trace analysis.
- The low perfect-oracle ceiling suggests a dataset or SAM-coverage limitation,
  not only a fusion-policy problem.

## Decision

Reject depth-decoupled fusion for MedQA.

Rationale:

- Best depth-decoupled gap is `+0.00%`, below the 3% gate.
- Perfect oracle ceiling is only `+1.45%`.
- SAM contributes too few useful candidates after EAGLE prefixes.

## Code Retention

Keep the analysis code. Do not implement a MedQA-specific runtime
depth-decoupled fusion mode from this result.

Experiment start commit: not recorded in the note.
Revert condition if discarded: not applicable, analysis-only result.

## Avoid Repetition

Do not repeat MedQA depth-decoupled oracle unless one of these changes:

- new Static SAM artifact with documented medical-domain provenance;
- new trace format that actually regenerates SAM from EAGLE leaves;
- different MedQA prompting format with a reason to expect higher SAM reuse.

## Next Step

Run the MT-Bench oracle gate from
`.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/plans/2026-06-10-depth-decoupled-oracle.md`.
