# Rank-Truncated OOV Probe Experiment Plan

**Goal:** Test whether corrected OOV token recovery survives explicit low-rank truncation of the ridge projection.

**Baseline:** Corrected shifted-index full-rank ridge, OOV ratio top-1/top-5/top-10 = 0.9108 / 0.9518 / 0.9636.

**Primary Metric:** OOV subset top-1 `ratio (pred/oracle)`; pass if rank 2780 or 2836 reaches at least 0.86.

**Budget:** One offline run on the existing medquad n=200 dump; no model inference and no training.

---

## Steps

1. Add rank sweep CLI.
   - File: `.codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
   - New option: `--rank-sweep 256,512,1024,1536,2048,2780,2836,3000,4096`
   - Keep existing full-rank output unchanged.

2. Refactor evaluation minimally.
   - Fit ridge `P` exactly as before.
   - Select the same test rows before scoring.
   - Reuse one `t2d` mask for all ranks.
   - Score full-rank first, then SVD-truncated `P_r`.

3. Run local cheap checks.
   - `python -m py_compile .codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
   - `git diff --check`

4. Run on server.
   - Use the exact n=200 dump, split seed 0, lambda 1.0, strict draft `t2d`.
   - Save log to `.codestable/brainstorms/spectral-vocab-recovery/runs/oov_recovery_rank_sweep_n200.log`.

5. Interpret.
   - If rank 2780/2836 OOV top-1 ratio >= 0.86, continue toward CCA-projected probe.
   - If only rank 4096 works, stop claiming low-rank spectral recovery.

## Command

```bash
python -u .codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py \
  --dump-dir evaluation/data/medquad/linear_alignment_probe_n200/dumps \
  --base-model-path /root/Models/Meta-Llama-3.1-8B-Instruct \
  --draft-model-path /root/Models/EAGLE3-LLaMA3.1-Instruct-8B \
  --lambda-reg 1.0 \
  --train-ratio 0.5 \
  --split-seed 0 \
  --top-ks 1,5,10 \
  --max-test-rows 8000 \
  --rank-sweep 256,512,1024,1536,2048,2780,2836,3000,4096 \
  2>&1 | tee .codestable/brainstorms/spectral-vocab-recovery/runs/oov_recovery_rank_sweep_n200.log
```

## Artifacts

- Server log: `.codestable/brainstorms/spectral-vocab-recovery/runs/oov_recovery_rank_sweep_n200.log`
- Updated result note after run: `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/2026-05-28-rank-truncated-oov-probe.md`

## Stop Conditions

- Stop if full-rank output no longer matches the known corrected baseline within normal sampling noise.
- Stop if SVD fails or rank 4096 does not approximately reconstruct full-rank behavior.
