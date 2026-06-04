# RerankSpec Oracle-Gap Pilot Result Summary

## Claim

- Tested an early, incorrectly framed RerankSpec Go/No-Go question: whether a synthetic Dynamic-SAM-derived candidate set showed an oracle-selection gap.
- Correction after code review: standard Dynamic SAM is a single-path proposer, so this run should be treated only as a collector/debug sanity check. It does **not** answer the real top-M tree-candidate oracle-gap question, which should use the SAM-only Static SAM tree proposer or another genuine multi-candidate proposer.
- This run does **not** test learned reranking, throughput, Static SAM tree candidates, Suffix-style retrieval, or serving performance.

## Evidence

- Baseline run: baseline candidate came from the first / longest-match continuation in the earlier Dynamic-SAM collector path.
- New run: oracle-best candidate among the collected synthetic candidates for each decode step.
- Important correction: these candidates are not the true SAM-only tree candidates; do not use this table as evidence against RerankSpec.
- Metrics pasted from the pilot summary:

| metric | value |
|---|---:|
| steps with SAM candidates | 7 |
| positive oracle-gap steps | 0 |
| positive oracle-gap rate | 0.0% |
| mean baseline accepted tokens | 1.857 |
| mean oracle accepted tokens | 1.857 |
| mean oracle gap | 0.000 |
| total oracle gap | 0 |
| mean candidate count | 2.000 |
| max oracle gap | 0 |

- Selection rule: continue toward acceptance-aware reranking only if oracle-best among top-M materially improves accepted length over the baseline selector.

## Confounders

- Only 7 steps had enough SAM candidates, so the sample is too small to reject the whole research direction.
- Mean candidate count is only 2.0, which means the current collector is not yet exercising a rich top-M candidate-selection setting.
- The run tested a Dynamic-SAM-derived path, but Dynamic SAM is fundamentally single-path in the standard SAM-only implementation; the observed candidate count of 2 came from the earlier synthetic tree/debug path, not the real SAM-only Static SAM tree proposer.
- It does not cover Static SAM tree candidates, SuffixDecoding, n-gram, REST-style retrieval, or hybrid EAGLE paths.
- No throughput, latency, candidate-enumeration overhead, or reranker overhead was measured.
- The exact model path, prompt set, and JSONL artifact path were not recorded in this note; add them if this result is used for a paper-facing comparison.

## Decision

- Decision: **debug / rerun before trusting**.
- Current evidence does **not** support building a learned reranker yet, but it also should not be interpreted as evidence against candidate selection: the proposer under test was not the intended top-M tree proposer.
- The next check should use `--candidate-source sam-only-static-tree` with an existing `samd_sam_only` Static SAM artifact and existing evaluation `question.jsonl` data.

## Code Retention

- Code retention: **discard from this project / migrate idea out**.
- Experiment branch: `rerankspec-oracle-gap`.
- Experiment start commit observed in this session: `ecbcb8c`.
- Working tree was already dirty before the branch switch, so any discard must restore only RerankSpec-touched files rather than using destructive rollback.
- Discard rationale: the user decided this idea should be completed in a different project better suited for multi-candidate speculative decoding experiments.
- Cleanup scope: remove RerankSpec pilot code/tests from this repository, archive the CodeStable brainstorm as migrated out, and keep this result note only as closeout evidence.

## Avoid Repetition

Do not repeat this exact one-prompt / Dynamic-SAM-derived setup expecting reranking evidence. Repeat only if at least one of the following changes is made:

- increase effective candidate diversity so `mean_candidate_count` is meaningfully above 2;
- run on a multi-prompt workload with enough SAM-hit steps;
- use a genuine multi-candidate proposer such as SAM-only Static SAM tree, n-gram, Suffix-style, or REST-style candidate sets;
- inspect per-step JSONL to verify whether tree leaf extraction is collapsing to near-duplicate paths.

## Next Step

Run a diagnostic rerun on the actual SAM-only Static SAM tree proposer with larger candidate diversity and more prompts:

1. keep `--top-m 8` or `--top-m 16`, but inspect whether the resulting `candidate_token_ids` are actually diverse;
2. lower or sweep `--len-threshold` only as a diagnostic, not as a paper setting;
3. use 20-100 repeated-context / agentic prompts;
4. if `positive_gap_rate` remains 0 and `mean_candidate_count` rises meaningfully, reject this Static-SAM-tree reranking path and test a different proposer.
