# Decoding-Step OOV Probe Experiment Plan

**Goal:** Test whether real decoding-step EAGLE3 draft hiddens recover strict OOV verifier-target signal through full-rank ridge and CCA-projected maps.

**Baseline:** Prompt first-pass dump-level CCA rank 2780 OOV top-1 ratio min/mean/max over split seeds 0-3 = 0.9371 / 0.9731 / 0.9964.

**Primary Metric:** OOV subset top-1 `ratio (pred/oracle)` on decoding-step dumps; pass if CCA rank 2780 or 2836 reaches at least 0.86 and oracle top-1 is non-trivial.

**Budget:** One small server sanity slice, then one n=200 MedQuAD dump and offline analysis. No training.

---

## Start State

- Branch: `feat/linear-alignment-probe`
- Start commit recorded locally before this change: `054a0ad9a2114ed215b52517d13f015782bcc8be`
- Tree status: dirty from accepted prompt OOV probe edits, CCA/rank docs, and this decoding-step probe implementation.
- Expected touched files:
  - `samd/utils.py`
  - `samd/samd_model.py`
  - `samd/tree_model/eagle3/eagle3.py`
  - `samd/tree_model/eagle3/eagle3_model.py`
  - `evaluation/eval_llama3.py`
  - `evaluation/inference_samd.py`
  - `.codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
  - `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/specs/2026-05-28-decoding-step-oov-probe.md`
  - `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/plans/2026-05-28-decoding-step-oov-probe.md`
  - `.codestable/brainstorms/spectral-vocab-recovery/brainstorm.md`

## Steps

1. Add an opt-in decoding-step capture path.
   - New config fields: `collect_decoding_step_probe`, `decoding_step_probe_dump_dir`, `decoding_step_probe_question_id`.
   - The default remains `False`; non-probe generation should not capture base hiddens or draft hiddens.

2. Capture strict aligned rows.
   - In EAGLE3 `topK_genrate`, when opt-in is enabled, expose normalized draft lm-head hiddens for compacted tree nodes.
   - In `SamdModel.decode`, for tree path only, record one row at `verifier_col = accept_length - 1` predicting `target_col = accept_length`.
   - Store `target_tokens` as the verifier argmax at that logit position, not as the draft rejected token.

3. Extend the evaluator.
   - Add `--dump-kind auto|prompt|decoding-step` to `.codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`.
   - Keep prompt mode's corrected shift: `h_d[:-2] -> h_b[1:-1]`, target `prompt_ids[2:]`.
   - In decoding-step mode, use stored `h_d`, `h_b`, and `target_tokens` directly.

4. Run local cheap checks.
   - `python -m py_compile samd/samd_model.py samd/utils.py samd/tree_model/eagle3/eagle3.py samd/tree_model/eagle3/eagle3_model.py evaluation/eval_llama3.py evaluation/inference_samd.py .codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
   - `git diff --check`

5. Server sanity dump.
   - Run a small question slice with `--collect_decoding_step_probe`.
   - Inspect the generated `.pt` files for row count and shape consistency before full collection.

6. Server full dump and analysis.
   - Collect n=200 MedQuAD decoding-step dumps.
   - Run `oov_recovery_probe.py --dump-kind decoding-step` with full-rank ridge, rank sweep, CCA sweep, strict `t2d`, and `--split-unit dump`.

7. Interpret.
   - If oracle top-1 is healthy and CCA rank 2780/2836 OOV top-1 ratio >= 0.86, the prompt first-pass result transfers to real decoding steps.
   - If full-rank passes but CCA fails, keep only a decoding-step-specific ridge/RRR route.
   - If oracle fails, inspect dump pairing before interpreting recovery.

## Artifacts

- Sanity dump dir: `evaluation/data/medquad/decoding_step_probe_sanity/dumps`
- Full dump dir: `evaluation/data/medquad/decoding_step_probe_n200/dumps`
- Analysis log: `.codestable/brainstorms/spectral-vocab-recovery/runs/oov_recovery_decoding_step_n200.log`
- Future result note after server run: `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/2026-05-28-decoding-step-oov-probe.md`

## Stop Conditions

- Stop if local `py_compile` or `git diff --check` fails.
- Stop after server sanity if dumps are empty or `h_d.shape[0] != h_b.shape[0] != target_tokens.shape[0]`.
- Stop if oracle top-1 is near zero; inspect alignment before running more seeds.
