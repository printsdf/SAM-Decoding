# Decoding-Step OOV Probe Experiment Spec

## Goal

Test whether the prompt first-pass OOV recovery result transfers to real SAM-Decoding tree-decoding steps.

The key difference from earlier prompt dumps is that rows must come from actual inference-time verifier positions: draft hidden used to predict a tree token, matching base verifier hidden/logit position, and the verifier target token id from that same position.

## Baseline To Beat

Prompt first-pass dump-level CCA result from `2026-05-28-cca-projected-oov-probe.md`:

- CCA rank 2780 dump-split seed 0 OOV ratio top-1/top-5/top-10: 0.9660 / 0.9662 / 0.9764
- CCA rank 2780 dump-split seeds 0-3 OOV top-1 ratio min/mean/max: 0.9371 / 0.9731 / 0.9964
- Full-rank ridge dump-split seed 0 OOV ratio top-1/top-5/top-10: 0.9388 / 0.9638 / 0.9657

## Metric

Primary metric: decoding-step OOV subset top-1 `ratio (pred/oracle)` under strict `t2d` target-token reachability.

Selection rule: CCA rank 2780 or 2836 should reach at least 0.86 OOV top-1 ratio under dump-level split. Full-rank ridge is a sanity baseline; if oracle top-1 is near zero, the dump pairing is suspect and no recovery claim should be made.

## Dump Schema

Each `.pt` dump uses `schema = "samd_decoding_step_probe_v1"` and contains:

- `h_d`: `[N, H]` draft hidden rows, each the EAGLE3 normalized lm-head input used to predict the verifier target position.
- `h_b`: `[N, H]` base verifier hidden rows from the same tree node/logit position.
- `target_tokens`: `[N]` verifier target token ids, computed as `argmax(candidate_logits[best_candidate, accept_length - 1])`.
- `records`: per-row metadata with `question_id`, `step_idx`, `decode_length_before_step`, `accept_length`, `best_candidate`, `verifier_logit_col`, `candidate_target_col`, `tree_node_index`, `first_rejected_token_id`, and `verifier_target_token_id`.
- `prompt_ids`: the prompt tokens for traceability only; the analysis does not use prompt-position shifted labels.

No prompt shift is applied to decoding-step rows: `h_d`, `h_b`, and `target_tokens` are already verifier-position aligned at dump time.

## Minimal Code Change

- Add opt-in `SamdGenerationConfig.collect_decoding_step_probe` and `decoding_step_probe_dump_dir`.
- In EAGLE3 only, optionally expose normalized draft hiddens for compacted tree nodes.
- In `SamdModel.decode`, record one strict row per real tree step: the position immediately before the first rejected draft token.
- Extend `oov_recovery_probe.py` with `--dump-kind {auto,prompt,decoding-step}` while preserving prompt shifted-index behavior.

Default generation remains unchanged because all new capture paths are gated by `collect_decoding_step_probe=False`.

## Sanity Checks

- `python -m py_compile samd/samd_model.py samd/utils.py samd/tree_model/eagle3/eagle3.py samd/tree_model/eagle3/eagle3_model.py evaluation/eval_llama3.py evaluation/inference_samd.py .codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
- `git diff --check`
- Server sanity run on a small question slice before collecting the full n=200 dump.
- Confirm generated dump files have non-empty `h_d`, `h_b`, `target_tokens` with matching first dimension.

## Failure Modes

- Empty or tiny decoding-step row count: the selected benchmark slice may hit sequence/SAM paths too often or accept-to-end steps; increase question count before interpreting.
- Low oracle top-1: base lm_head is not reproducing verifier targets from the paired `h_b`; treat as a pairing bug until inspected.
- Full-rank ridge passes but CCA rank 2780 fails: prompt first-pass CCA subspace does not directly transfer; keep spectral route only with decoding-step-specific map or RRR fallback.
- Both full-rank and CCA fail while oracle is healthy: draft hidden does not retain recoverable decoding-step OOV verifier signal.

## Next Decision

If decoding-step CCA rank 2780/2836 passes, proceed to end-to-end accept-length / wall-time intervention. If it fails, do not claim paper-ready decoding-step recovery from prompt first-pass evidence.
