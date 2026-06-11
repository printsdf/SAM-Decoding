# Rank-Truncated OOV Probe Experiment Spec

## Goal

Test whether corrected OOV token recovery survives explicit low-rank truncation of the ridge projection.

## Baseline To Beat

Corrected shifted-index full-rank ridge from `2026-05-28-spectral-vocab-recovery-shifted-rerun.md`:

- OOV ratio top-1/top-5/top-10: 0.9108 / 0.9518 / 0.9636
- OOV hit_pred top-1/top-5/top-10: 0.3967 / 0.5479 / 0.6241
- Split: medquad n=200 dump, split-seed 0, n_train=14690, max-test-rows=8000, lambda=1.0

## Metric

Primary metric: OOV subset `ratio (pred/oracle)` top-1.

Selection rule: smallest rank whose OOV top-1 ratio is at least 0.86 and whose top-5/top-10 ratios do not collapse relative to full-rank.

## Split Assumptions

Use the exact same shifted-index loader, split seed, train ratio, max-test-rows, top-k set, base lm_head, and strict `t2d` OOV split as the full-rank rerun.

## Minimal Code Change

Add a `--rank-sweep` option to `.codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`.

The sweep truncates the fitted ridge matrix `P` via SVD:

```text
P_r = U[:, :r] diag(S[:r]) Vh[:r, :]
```

and evaluates `h_d_test @ P_r + h_b_mean` on the same test rows and OOV masks as full-rank.

## Sanity Checks

- `python -m py_compile .codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
- One server run with `--rank-sweep 256,512,1024,1536,2048,2780,2836,3000,4096`
- Confirm rank 4096 approximately matches the full-rank baseline.

## Failure Modes

- If only rank 4096 matches full-rank, the evidence supports full-rank linear recovery but not a low-rank spectral method.
- If rank 2780/2836 keeps OOV top-1 ratio >= 0.86, the spectral subspace route remains viable.
- If all truncated ranks fail badly, do not claim Subspace Identity Projection.

## Next Decision

If rank 2780/2836 passes, implement a CCA-projected variant and then test decoding-step hidden recovery. If it fails, reframe the method away from low-rank subspace recovery.
