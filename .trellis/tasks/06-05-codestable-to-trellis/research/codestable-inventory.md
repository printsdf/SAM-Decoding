# CodeStable Inventory and Migration Mapping

## Source Summary

`.codestable/` files inspected before deletion:

* `attention.md` - startup-critical project notes.
* `architecture/ARCHITECTURE.md` - SAM-Decoding module map and hard constraints.
* `requirements/VISION.md` plus requirement files for evaluation V_miss diagnosis, SAM/EAGLE3 tree fusion, and Static SAM offline corpus.
* `compound/*.md` - active decisions, learnings, and one EAGLE3 compatibility exploration.
* `features/*` - historical design/checklist/acceptance artifacts.
* `reference/*` and `tools/*` - CodeStable workflow mechanics and utilities.

## Migration Mapping

| CodeStable Source | Trellis Destination | Notes |
| --- | --- | --- |
| `.codestable/attention.md` | `.trellis/spec/backend/runtime-constraints.md`, `.trellis/spec/backend/quality-guidelines.md` | Import startup-critical package, tokenizer, script, and EAGLE3 weight constraints. |
| `.codestable/architecture/ARCHITECTURE.md` | `.trellis/spec/backend/project-architecture.md` | Convert module map and hard boundaries into Trellis architecture guidance. |
| `.codestable/requirements/*.md` | `.trellis/spec/backend/project-capabilities.md` | Preserve current/draft capability intent and scope boundaries. |
| Active `.codestable/compound/*decision*.md` | `.trellis/spec/backend/runtime-constraints.md`, `.trellis/spec/backend/eagle3-integration.md`, `.trellis/spec/backend/evaluation-protocols.md`, `.trellis/spec/backend/tree-fusion.md` | Convert active decisions into enforceable project contracts. |
| Active `.codestable/compound/*learning*.md` | `.trellis/spec/backend/eagle3-integration.md`, `.trellis/spec/backend/evaluation-protocols.md`, `.trellis/spec/backend/quality-guidelines.md` | Convert repeated pitfalls and checklists into implementation/review guidance. |
| `.codestable/compound/2026-05-23-explore-eagle3-compatibility.md` | `.trellis/spec/backend/eagle3-integration.md` | Use as supporting source for EAGLE variant integration dimensions. |
| `.codestable/features/*` | Migration inventory only | Source evidence was inspected before deletion; do not convert each historical feature into a Trellis task in this MVP. |
| `.codestable/reference/*`, `.codestable/tools/*` | No direct migration | These describe CodeStable's old workflow, not SAM-Decoding project rules. |

## Hard Constraints Extracted

* SAM-Decoding depends on `torch`, `transformers` 4.46.x, and `safetensors`; missing packages fail at `samd/__init__.py` import time.
* `transformers` must stay on 4.46.x unless the Llama monkey patches are deliberately rewritten.
* `SamdModel` supports `batch_size == 1` only.
* `tree_method="eagle3"` must install the EAGLE3-specific base model patch, otherwise EAGLE3 gets the wrong hidden-state shape.
* EAGLE3 hidden-state capture indices are fixed to `{2, N//2, N-3}` and are not user-configurable.
* Official EAGLE3 weights omit `embed_tokens.weight`; the integration copies base-model embeddings after weight load when vocab sizes match.
* Llama-3.1 and long-context models need explicit `--max_cache_len <= 4096` for `evaluation/inference_samd.py` and `evaluation/inference_sam_only.py` on 24 GiB GPUs.
* Benchmark scripts that report both wall-time speedup and V_miss must split trace-OFF speed measurement from trace-ON diagnosis collection.
* Current SAM + EAGLE3 default fusion is `tree_method="eagle3"` with `tree_fusion="sam_sequence_graft"`.
* Llama-3 tokenizer paths must preserve the `pad_token = eos_token` fallback before using `padding=True`.

## Historical Archive

The old `.codestable/` directory was removed after migration at the user's request. This inventory is the durable trace of what existed and where the long-lived knowledge was moved. Trellis specs under `.trellis/spec/backend/` are now the active source of truth.
