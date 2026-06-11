# CCA-Projected OOV Probe Experiment Plan

**Goal:** Test whether CCA shared directions directly recover OOV token signal.

**Baseline:** Rank-truncated ridge, especially rank 1024 OOV top-1 ratio 0.8981 and rank 2780/2836 OOV top-1 ratio 0.9108.

**Primary Metric:** OOV subset top-1 `ratio (pred/oracle)`; pass if CCA rank 2780 or 2836 reaches at least 0.86.

**Budget:** One offline run on the existing medquad n=200 dump; no model inference and no training.

---

## Start State

- Branch: `feat/linear-alignment-probe`
- Commit before this code change: `054a0ad9a2114ed215b52517d13f015782bcc8be`
- Tree status: dirty from the accepted shifted-index/rank-sweep probe edits and result docs.
- Touched files expected for this experiment:
  - `.codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
  - `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/specs/2026-05-28-cca-projected-oov-probe.md`
  - `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/plans/2026-05-28-cca-projected-oov-probe.md`

## Steps

1. Add CCA sweep CLI.
   - File: `.codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
   - New options: `--cca-sweep` and `--cca-niter`
   - Keep existing full-rank and rank-sweep outputs unchanged.

2. Implement CCA-regression reconstruction.
   - Compute randomized SVDs of `h_d_train_c` and `h_b_train_c` with `q=max(cca_sweep)`.
   - Compute CCA via SVD of `U_d.T @ U_b`.
   - Reconstruct `h_b` using top-r canonical directions:
     `Y_hat_r = (X_test Vx Sx^-1 A_r diag(rho_r)) B_r^T Sy Vy^T + h_b_mean`.

3. Run local cheap checks.
   - `python -m py_compile .codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
   - `git diff --check`

4. Run on server.
   - Same n=200 dump, split seed 0, lambda 1.0, strict draft `t2d`.
   - Save log to `.codestable/brainstorms/spectral-vocab-recovery/runs/oov_recovery_cca_sweep_n200.log`.

5. Interpret.
   - If rank 2780/2836 OOV top-1 ratio >= 0.86, CCA shared basis is viable.
   - If it fails while rank-truncated ridge passes, keep the method as RRR/ridge-SVD and use CCA as diagnostic evidence only.

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
  --cca-sweep 256,512,1024,1536,2048,2780,2836,3000 \
  --cca-niter 10 \
  2>&1 | tee .codestable/brainstorms/spectral-vocab-recovery/runs/oov_recovery_cca_sweep_n200.log
```

## Artifacts

- Server log: `.codestable/brainstorms/spectral-vocab-recovery/runs/oov_recovery_cca_sweep_n200.log`
- Result note after run: `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/results/2026-05-28-cca-projected-oov-probe.md`

## Stop Conditions

- Stop if full-rank output no longer matches the known corrected baseline.
- Stop if CCA SVD fails before scoring; do not interpret partial output.
