# Directory Structure

> How backend/research code is organized in this project.

## Overview

SAM-Decoding code is organized by runtime responsibility, not by web-service
layers. Keep new code close to the subsystem it extends, and avoid creating
generic utility modules unless the same behavior is already needed in multiple
places.

## Directory Layout

```text
samd/
  samd_model.py              Main SAMD generation loop and verifier orchestration
  draft.py                   Draft source selection and tree-fusion orchestration
  samd_config.py             Runtime config for tree_method and tree_fusion
  diagnosis.py               Per-step diagnosis trace helpers
  cache.py                   Static cache implementation used by SAMD
  inference/cli.py           Library-style CLI inference entry
  model_patch/               Llama monkey patches for tree attention / EAGLE3
  sam/                       Dynamic and Static SAM implementations
  tree_model/
    fusion.py                TreeSpec conversion/export and SAM graft/merge logic
    token_recycle/           Token Recycle draft model integration
    eagle/                   EAGLE-1 integration
    eagle2/                  EAGLE-2 integration
    eagle3/                  EAGLE-3 integration and optional tail sidecar

samd_sam_only/               SAM-only optimized path, independent from tree_method
evaluation/                  Spec-Bench/MT-Bench/MedQA/MedQuAD inference and analysis
evaluation/model/            Vendored or legacy baseline model implementations
scripts/                     Shell entry points for smoke tests and benchmark batches
tools/                       Offline Static SAM corpus/artifact tools
tests/                       Pytest and executable integration checks
docs/                        Usage guides and experiment specs/results
```

## Module Organization

* Put draft-model-specific code under `samd/tree_model/<method>/`.
* Put candidate-tree combination code in `samd/tree_model/fusion.py` unless it
  belongs entirely to one method.
* Put base-model monkey patches under `samd/model_patch/`; patch selection must
  remain explicit in `SamdModel.register_forward_patch`.
* Put reusable evaluation logic under `evaluation/`; put runnable batch recipes
  under `scripts/`.
* Put Static SAM corpus and artifact builders under `tools/static_sam/`; keep
  compatibility wrappers such as `tools/gen_sam_alpaca.py` thin.
* Do not add frontend specs or frontend files for this project unless a real UI
  is introduced.

## Naming Conventions

* Runtime config fields use snake_case and should match CLI flags where
  possible, for example `tree_method`, `tree_fusion`, and `max_cache_len`.
* `tree_method` selects the draft model family (`token_recycle`, `eagle`,
  `eagle2`, `eagle3`).
* `tree_fusion` selects candidate-combination strategy (`none`,
  `sam_sequence_graft`, `sam_tree_union_prune`, `eagle_prefix_sam_expand`).
* Do not encode fusion variants as new `tree_method` values.
* Experiment docs should stay under `docs/experiments/specs/` and
  `docs/experiments/results/` with date-prefixed filenames.

## Examples

* `samd/tree_model/eagle3/` is the model-specific home for EAGLE3 architecture,
  weights, vocabulary mapping, and incremental state handling.
* `samd/tree_model/fusion.py` is the shared home for tree representation and
  export logic used by multiple fusion strategies.
* `scripts/run_bench_cross_domain_speedup.sh` is the reference pattern for a
  single shell script that fetches data, runs benchmark phases, analyzes
  outputs, and reports progress.
