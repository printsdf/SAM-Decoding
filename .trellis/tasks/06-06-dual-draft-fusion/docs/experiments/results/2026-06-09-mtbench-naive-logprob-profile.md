# MT-Bench Naive Logprob And Profiling Result Summary

## Claim

Phase 3.1 EAGLE3 logprob extraction made the naive fusion result comparable to
pure EAGLE3, but it did not beat EAGLE3-only or `sam_sequence_graft` on the
current MT-Bench evidence. A separate tail-enabled profiling run reported a
`+2.91%` oracle gap, but that run is not a valid no-tail decision gate.

## Evidence

Current full MT-Bench comparison:

| Method | Rows | Turns | MAT | TPS | vs Eagle3 TPS | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pure_eagle3` | 80 | 160 | 6.326 | 55.917 | 1.000x | 9976 |
| `sam_sequence_graft` | 80 | 160 | 6.627 | 56.211 | 1.005x | 9557 |
| `naive_logprob` | 80 | 160 | 6.310 | 53.539 | 0.957x | 10022 |

Current-machine subset rerun on host `cs-01ktjzagnfptxzvk7zzs604xxk`, NVIDIA
L4 23034 MiB, same MT-Bench question IDs 81..91:

| Method | Rows | Turns | MAT | TPS | vs Eagle3 | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pure_eagle3_current_q0_11` | 11 | 22 | 6.189 | 54.109 | 1.000x | 1129 |
| `naive_logprob_current_q0_11` | 11 | 22 | 6.075 | 52.047 | 0.962x | 1144 |

Tail-enabled profiling/oracle run on 2026-06-09:

| Metric | Value |
| --- | ---: |
| decode steps | 1783 |
| fusion overhead | 0.52% |
| fusion logic time | 1.055979 s |
| total time | 205.702146 s |
| oracle gap vs `eagle3_only` | +2.91% |

Tail-enabled oracle table:

| Method | MAT | Nodes | Oracle gap |
| --- | ---: | ---: | ---: |
| `eagle3_only` | 3.0039 | 59.00 | +0.00% |
| `sam_sequence_graft` | 3.0118 | 59.08 | +0.26% |
| `perfect` | 3.0914 | 59.08 | +2.91% |
| `budgeted` | 3.0914 | 59.08 | +2.91% |
| `source_balanced` | 3.0914 | 59.08 | +2.91% |

Remote test evidence:

```text
tests/test_naive_fusion.py: 11 passed in 5.02s
```

## Confounders

- The tail-enabled profiling run had `EAGLE3_TAIL_PATH=./tail_epoch_10.pt` and
  `EAGLE3_TAIL_TYPE=auto` from `.env`; it cannot decide the no-tail gate.
- Earlier 140+ TPS baselines were not proven to share the same machine/runtime
  provenance as the 50+ TPS current-machine runs.
- The repository was dirty during validation. The pure `09c97b5` archive was
  not self-contained because some EAGLE3 tail and naive-fusion API changes were
  outside that commit.
- Trace-on and fusion-profile outputs are diagnostics, not speedup baselines.

## Decision

Do not claim Phase 3.1 improved MT-Bench throughput or MAT. Treat
`naive_logprob` as a negative/neutral baseline.

Do not use the `+2.91%` tail-enabled oracle as the no-tail payoff-aware or
depth-decoupled decision gate.

## Code Retention

Keep the logprob and profiling work as diagnostic infrastructure unless a later
cleanup task decides otherwise.

Experiment start commit: `09c97b5` was referenced but not self-contained.
Revert condition if discarded: only after confirming no later experiments depend
on EAGLE3 logprob metadata or fusion profile traces.

## Avoid Repetition

Do not rerun this comparison without recording:

- machine hostname and GPU;
- whether EAGLE3 tail sidecar is enabled;
- `max_cache_len`, `max_new_tokens`, EAGLE3 budget, and question IDs;
- trace-off versus trace-on status.

## Next Step

Run the no-tail MT-Bench depth-decoupled oracle gate and record a new result
note under `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/`.
