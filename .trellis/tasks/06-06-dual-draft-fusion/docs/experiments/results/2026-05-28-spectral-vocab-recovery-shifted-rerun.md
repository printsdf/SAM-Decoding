# Spectral Vocab Recovery Shifted Rerun Result Summary

## Claim

- Tested whether the OOV recovery result survives the corrected EAGLE3 shift alignment.
- Answer: yes for full-rank ridge token-level recovery. Follow-up rank-truncated ridge also passed; CCA-projected spectral method remains untested.

## Evidence

- Baseline / invalid run: old OOV probe used `h_d[:-1] -> h_b[:-1]`, target=`prompt_ids[1:]`, which could leak token `i+1` into `h_d[i]`.
- New run: server rerun after confirming `oov_recovery_probe.py` contains `h_d_parts.append(h_d[:-2])`.
- Split: medquad n=200 dump, n_train=14690, n_test eval=8000, lambda=1.0, strict t2d OOV split.
- Global ratio top-1/top-5/top-10: 0.9583 / 0.9720 / 0.9706.
- In-vocab ratio top-1/top-5/top-10: 0.9635 / 0.9739 / 0.9713 (n=7279).
- OOV ratio top-1/top-5/top-10: 0.9108 / 0.9518 / 0.9636 (n=721).
- OOV absolute hit_pred top-1/top-5/top-10: 0.3967 / 0.5479 / 0.6241.
- OOV oracle top-1/top-5/top-10: 0.4355 / 0.5756 / 0.6477.
- CCA rerun: q=3000, niter=10, seed=0, rho[1..2000]=1.0000, rho < 0.99 at rank 2782, rho < 0.01 at rank 2836.

## Confounders

- This is prompt first-pass hidden recovery, not decoding-step recovery.
- The tested method is full-rank ridge with shrinkage, not rank-r RRR, CCA projection, or Subspace Identity Projection.
- No seed sweep yet; the result uses split-seed 0.

## Decision

- Keep and extend.
- The corrected run clears the OOV token-level recovery go/no-go threshold.
- Do not claim paper-ready spectral recovery until a CCA-projected variant and decoding-step recovery are tested.

## Code Retention

- Keep code changes to shifted indexing and clarified docs.
- Experiment start commit: not recorded.
- Revert condition if discarded: only if a later direct rerun with the same corrected script fails to reproduce OOV ratio > 0.7.

## Avoid Repetition

- Do not rerun the old `h_d[:-1] -> h_b[:-1]`, target=`prompt_ids[1:]` probe as evidence.
- Do not treat full-rank ridge λ=1.0 as proof of a 2780-dimensional subspace method.

## Next Step

- Rank-truncated ridge has now passed; next implement a CCA-projected OOV probe and compare it against the full-rank and rank-truncated results above.
