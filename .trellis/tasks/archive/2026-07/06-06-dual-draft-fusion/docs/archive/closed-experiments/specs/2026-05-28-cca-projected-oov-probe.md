# CCA-Projected OOV Probe Experiment Spec

## Goal

Test whether OOV recovery works using CCA shared directions directly, rather than SVD truncation of the fitted ridge projection `P`.

## Baseline To Beat

Rank-truncated ridge from `2026-05-28-rank-truncated-oov-probe.md`:

- Rank 1024 OOV ratio top-1/top-5/top-10: 0.8981 / 0.9325 / 0.9529
- Rank 2780 OOV ratio top-1/top-5/top-10: 0.9108 / 0.9470 / 0.9615
- Full-rank OOV ratio top-1/top-5/top-10: 0.9108 / 0.9518 / 0.9636

## Metric

Primary metric: OOV subset `ratio (pred/oracle)` top-1.

Selection rule: CCA rank 2780 or 2836 should reach at least 0.86 top-1 ratio. Rank 1024 is considered strong if it remains close to the rank-truncated ridge 1024 result.

## Split Assumptions

Use the exact same shifted-index loader, split seed, train ratio, max-test-rows, top-k set, base lm_head, and strict `t2d` OOV split as the previous full-rank and rank-truncated reruns.

## Minimal Code Change

Add a `--cca-sweep` option to `.codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`.

The CCA mapping uses train split SVDs:

```text
X = Ux Sx Vx^T
Y = Uy Sy Vy^T
Ux^T Uy = A diag(rho) B^T
Y_hat_r = (X_test Vx Sx^-1 A_r diag(rho_r)) B_r^T Sy Vy^T
```

This differs from `--rank-sweep`, which truncates the already fitted ridge projection.

## Sanity Checks

- `python -m py_compile .codestable/brainstorms/spectral-vocab-recovery/oov_recovery_probe.py`
- One server run with `--cca-sweep 256,512,1024,1536,2048,2780,2836,3000`
- Confirm the full-rank ridge baseline printed by the same command still matches previous results.

## Failure Modes

- If CCA projection fails while rank-truncated ridge passes, the publishable method is RRR/ridge-SVD rather than pure CCA shared basis.
- If CCA projection passes at rank 2780/2836, the Subspace/CCA narrative is much stronger.
- If CCA projection passes at rank 1024, rank-cost tradeoff may be better than the CCA knee suggests.

## Next Decision

If CCA projection passes, run a decoding-step hidden recovery probe. If it fails, proceed with rank-truncated ridge as the method and use CCA only as diagnostic evidence.
