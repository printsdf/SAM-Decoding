# Rank-Truncated OOV Probe Result Summary

## Claim

- Tested whether corrected OOV recovery depends on full-rank ridge.
- Answer: no. SVD-truncated ridge preserves OOV token-level signal; rank 1024 already passes the top-1 threshold, and rank 2780/2836 essentially matches full-rank top-1.

## Evidence

- Baseline: corrected shifted-index full-rank ridge from the same script, same split, same test subsample, same OOV mask.
- Dataset/split: medquad n=200 dump, n_train=14690, n_test eval=8000, lambda=1.0, split-seed=0, strict `t2d` OOV split.
- Full-rank OOV ratio top-1/top-5/top-10: 0.9108 / 0.9518 / 0.9636.
- Rank 256 OOV ratio: 0.6274 / 0.6072 / 0.6552.
- Rank 512 OOV ratio: 0.6688 / 0.9157 / 0.8158.
- Rank 1024 OOV ratio: 0.8981 / 0.9325 / 0.9529.
- Rank 1536 OOV ratio: 0.9045 / 0.9422 / 0.9572.
- Rank 2048 OOV ratio: 0.9076 / 0.9446 / 0.9593.
- Rank 2780 OOV ratio: 0.9108 / 0.9470 / 0.9615.
- Rank 2836 OOV ratio: 0.9108 / 0.9470 / 0.9615.
- Rank 4096 OOV ratio: 0.9108 / 0.9518 / 0.9636.

## Confounders

- This tests SVD truncation of the fitted ridge projection `P`; it does not yet test a pure CCA-projected basis or Subspace Identity Projection.
- This remains prompt first-pass recovery, not decoding-step recovery.
- Only split-seed 0 was run.
- Wall-clock and memory cost of applying rank 1024/2780 at inference were not measured.

## Decision

- Keep and extend.
- Rank-truncated ridge passes the planned success rule: rank 2780/2836 reaches OOV top-1 ratio 0.9108, and rank 1024 already exceeds the 0.86 threshold.
- This supports the spectral recovery route, but not yet a pure CCA basis method.

## Code Retention

- Keep `--rank-sweep` support in `oov_recovery_probe.py`.
- Experiment start commit: not recorded.
- Revert condition if discarded: only if a direct rerun with the same corrected script fails to reproduce rank 1024 OOV top-1 ratio > 0.86.

## Avoid Repetition

- Do not claim full-rank ridge is required for OOV recovery.
- Do not claim Subspace Identity Projection until a CCA-projected or identity-basis variant is tested.

## Next Step

- Implement CCA-projected OOV probe on the same shifted-index split to separate “SVD-truncated fitted P works” from “CCA shared basis works.”
