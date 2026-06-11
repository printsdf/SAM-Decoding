# CCA-Projected OOV Probe Result Summary

## Claim

- Tested whether train-only CCA shared directions can recover OOV next-token signal, instead of truncating the fitted ridge matrix `P`.
- Answer after the stricter dump-level split and seed robustness check: yes. CCA rank 2780/2836/3000 preserves most OOV oracle signal and beats the planned 0.86 top-1 ratio threshold even when whole prompts are split across train/test.

## Evidence

- Row-level baseline: corrected shifted-index full-rank ridge from the same command, same row-level split, same test subsample, same strict `t2d` OOV mask.
- Row-level dataset/split: medquad n=200 dump, n_train=14690, n_test eval=8000, lambda=1.0, split-seed=0, `--split-unit row`, strict `t2d` OOV split.
- Full-rank ridge OOV ratio top-1/top-5/top-10: 0.9108 / 0.9518 / 0.9636.
- Rank-truncated ridge 2780/2836 OOV ratio: 0.9108 / 0.9470 / 0.9615.
- CCA rank 256 OOV ratio: 0.0159 / 0.1711 / 0.1585.
- CCA rank 512 OOV ratio: 0.2197 / 0.3229 / 0.3833.
- CCA rank 1024 OOV ratio: 0.4045 / 0.4072 / 0.3640.
- CCA rank 1536 OOV ratio: 0.4013 / 0.4120 / 0.3704.
- CCA rank 2048 OOV ratio: 0.4172 / 0.5518 / 0.5503.
- CCA rank 2780 OOV ratio: 0.9873 / 0.9855 / 0.9872.
- CCA rank 2836 OOV ratio: 0.9873 / 0.9855 / 0.9872.
- CCA rank 3000 OOV ratio: 0.9873 / 0.9855 / 0.9872.
- CCA rank 2780 absolute OOV hit_pred top-1/top-5/top-10: 0.4300 / 0.5673 / 0.6394.
- OOV oracle top-1/top-5/top-10: 0.4355 / 0.5756 / 0.6477.

Dump-level split follow-up:

- Split: medquad n=200 dump, train_dumps=100, test_dumps=100, n_train=14650, n_test eval=8000, lambda=1.0, split-seed=0, `--split-unit dump`, strict `t2d` OOV split.
- Full-rank ridge OOV ratio top-1/top-5/top-10: 0.9388 / 0.9638 / 0.9657.
- Rank-truncated ridge rank 1024 OOV ratio: 0.9388 / 0.9444 / 0.8455.
- Rank-truncated ridge rank 2780/2836 OOV ratio: 0.9354 / 0.9589 / 0.9657.
- CCA rank 1024 OOV ratio: 0.1837 / 0.4082 / 0.4099.
- CCA rank 2048 OOV ratio: 0.1803 / 0.6449 / 0.5751.
- CCA rank 2780/2836/3000 OOV ratio: 0.9660 / 0.9662 / 0.9764.
- CCA rank 2780 absolute OOV hit_pred top-1/top-5/top-10: 0.4075 / 0.5739 / 0.6528.
- Dump-level OOV oracle top-1/top-5/top-10: 0.4218 / 0.5940 / 0.6686.

Dump-level split-seed robustness:

- Seed 0, CCA rank 2780/2836 OOV ratio: 0.9660 / 0.9662 / 0.9764.
- Seed 1, CCA rank 2780/2836/3000 OOV ratio: 0.9964 / 0.9975 / 0.9978.
- Seed 2, CCA rank 2780/2836/3000 OOV ratio: 0.9930 / 0.9853 / 0.9957.
- Seed 3, CCA rank 2780 OOV ratio: 0.9371 / 0.9690 / 0.9767.
- Seed 3, CCA rank 2836/3000 OOV ratio: 0.9371 / 0.9714 / 0.9767.
- Rank 2780 OOV top-1 ratio over seeds 0-3: min 0.9371, mean 0.9731, max 0.9964.
- CCA rank 2048 remains unstable/weak on OOV top-1 over seeds 1-3: 0.3891 / 0.3798 / 0.3642.

## Confounders

- This is still prompt first-pass hidden recovery, not decoding-step recovery.
- The row-level leakage concern was tested with `--split-unit dump` and did not explain away the result.
- Dump-level split-seed robustness passed for seeds 0-3, but all runs are still on the same medquad n=200 dump and model pair.
- The method uses train split base hiddens to fit the CCA map, but does not use test base hiddens or test labels in the reconstruction.
- Wall-clock, memory, and end-to-end acceptance impact were not measured.

## Decision

- Keep and extend.
- CCA-projected recovery passes the planned success rule at the spectral knee under both row-level and dump-level splits.
- Dump-level rank 2780/2836 OOV top-1 ratio is 0.9660, above the 0.86 threshold and above full-rank ridge's 0.9388 ratio on the same split.
- Low CCA ranks remain poor: rank 1024 OOV top-1 ratio is only 0.1837 under dump split. The useful dimension is near the observed CCA knee, not 1024, for pure CCA reconstruction.
- The spectral method route is now stronger than rank-truncated ridge alone, but the claim remains provisional until decoding-step probes pass.

## Code Retention

- Keep `--cca-sweep` support in `oov_recovery_probe.py`.
- Keep the new `--split-unit dump` option; it caught the intended stricter split and the result passed.
- Experiment start commit: `054a0ad9a2114ed215b52517d13f015782bcc8be`.
- Revert condition if discarded: only if decoding-step probes show the rank 2780/2836 CCA result does not generalize beyond this prompt first-pass setup.

## Avoid Repetition

- Do not claim that CCA rank 1024 is sufficient; it fails OOV top-1 on this run.
- Do not claim end-to-end acceleration or decoding-step recovery from this prompt-first-pass probe.
- Do not compare CCA rank 2780 against old unshifted OOV numbers.
- Do not describe the result as a row-split artifact; dump-level split passed.

## Next Step

- Run a decoding-step dump probe with verifier-target pairing.
