# Quality Guidelines

> Code quality standards for SAM-Decoding backend/research code.

## Overview

Prioritize reproducibility, explicit contracts, and preservation of verifier
semantics. This project often ports model code from EAGLE-family repositories,
so shape/calling-convention correctness matters more than superficial cleanup.

## Required Patterns

* Search before changing constants, enum values, CLI flags, or config fields.
* Keep `tree_method` and `tree_fusion` as separate dimensions.
* Preserve target-model greedy decoding semantics; candidate changes must only
  change what the verifier sees, not how final tokens are accepted.
* Keep diagnosis trace collection default-off and avoid adding tensor work on
  the normal path.
* When adding benchmark scripts, make them runnable end-to-end from one shell
  entry point and include analysis steps.
* Keep tests and docs aligned with CLI flags whenever new runtime options are
  added.

## Forbidden Patterns

* Do not silently support non-Llama backbones in `samd` patches. Current patches
  are Llama-specific.
* Do not broaden `batch_size` support locally without redesigning masks, cache
  slicing, SAM lookup, and acceptance logic.
* Do not upgrade `transformers` beyond 4.46.x without a deliberate rewrite and
  review of the monkey patches.
* Do not remove EAGLE3 `stable_kv` incremental behavior to make a local shape
  mismatch disappear.
* Do not reuse benchmark outputs across trace phases unless the protocol
  explicitly allows it, such as sam-only N/A rows in phase2 analysis.

## Testing Requirements

Use focused verification based on the touched area:

* Config/CLI changes: update parser/config tests or smoke commands that exercise
  the new flag.
* EAGLE3 changes: run EAGLE3-specific tests or equivalence checks when
  dependencies and model paths are available.
* Diagnosis trace changes: run `tests/test_samd_diagnosis.py` or a matching
  smoke script.
* Static SAM tooling changes: run `tests/test_static_sam_pipeline.py`.
* Benchmark-script-only changes: run shell syntax checks where possible and
  inspect commands for phase/output consistency.

Many tests import `torch`, `transformers`, and `safetensors`; if the local
environment lacks them, record that verification was not runnable rather than
pretending tests passed.

## Code Review Checklist

* Did the change preserve `batch_size == 1` assumptions?
* Did it keep `transformers` 4.46.x monkey-patch compatibility?
* Did it preserve EAGLE3 hidden-state shape and incremental state contracts?
* Did it avoid adding overhead to the default non-diagnosis generation path?
* Did it keep benchmark phase outputs separate and named clearly?
* Did it update Trellis specs if a new durable rule was discovered?
