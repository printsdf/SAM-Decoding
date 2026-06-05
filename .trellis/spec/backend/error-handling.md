# Error Handling

> How failures should be handled in SAM-Decoding code and scripts.

## Overview

This is research/runtime code, not an API server. Prefer explicit fail-fast
checks for invalid runtime configurations and shape contracts. Use warnings only
when the code can continue but the output may be unusable and the warning is
more useful than aborting construction.

## Error Types

* Use `ValueError` for invalid user-facing configuration combinations, such as
  unsupported `tree_fusion` values or fusion strategies used with a non-EAGLE3
  `tree_method`.
* Use `assert` for internal invariants already assumed throughout the model
  path, such as `batch_size == 1`.
* Use printed warnings in CLI/script contexts for malformed optional analysis
  records that can be skipped without invalidating the whole file.

## Required Patterns

* Validate config combinations in config objects or construction paths before
  entering the hot generation loop.
* Prefer a hard failure for shape mismatches that would otherwise produce
  silent draft-quality regressions.
* Preserve original third-party calling conventions when porting model code;
  investigate shape mismatches against the upstream caller before "fixing"
  vendored logic.
* Keep diagnosis trace parsing tolerant: malformed trace steps should warn and
  be skipped, while a completely missing trace should warn that the input was
  probably produced without `--collect_diagnosis_trace`.

## Warning Cases

The EAGLE3 integration may continue with a warning when base and draft
`embed_tokens` shapes differ. Construction can technically finish, but draft
outputs are meaningless because the official EAGLE3 design assumes shared base
embedding space. Treat this warning as a signal to stop the run unless the task
is explicitly testing failure behavior.

## Common Mistakes

* Converting an intentional fail-fast shape error into a fallback path, then
  measuring a silent garbage draft model.
* Removing EAGLE3 `stable_kv` incremental slicing to satisfy a local length
  mismatch without checking the original EAGLE caller's delta-hidden-state
  convention.
* Letting `tree_fusion` silently no-op for unsupported `tree_method` values.
