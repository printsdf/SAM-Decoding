# Project Architecture

> Current architecture map for SAM-Decoding.

## Purpose

SAM-Decoding accelerates autoregressive generation by using suffix-automaton
matches from the prompt or an offline text base as draft candidates. It can also
combine SAM with draft-model-based speculative decoding methods such as EAGLE,
EAGLE2, EAGLE3, and Token Recycle.

## Core Runtime Flow

1. `SamdModel.generate` owns the decode loop and target-model verification.
2. `DraftModel.lookup` selects or combines draft sources based on
   `SamdConfig`.
3. SAM paths generate sequence candidates from suffix matches.
4. Tree-model paths generate tree candidates from Token Recycle/EAGLE-family
   draft models.
5. Optional fusion converts multiple candidate sources into standard tree
   buffers.
6. The target model verifies candidates; only verifier-accepted tokens become
   output.

## Main Subsystems

### `samd/`

Primary SAM-Decoding implementation.

* `samd_model.py` - generation loop, forward patch registration, cache setup,
  candidate verification, accept-length accounting, and optional diagnosis trace
  collection.
* `draft.py` - draft source orchestration and fusion-mode dispatch.
* `samd_config.py` - configuration object for tree methods, fusion modes, and
  fusion budgets.
* `diagnosis.py` - pure helper functions for per-step V_miss/acceptance traces.
* `cache.py` - static KV cache implementation.

### `samd/model_patch/`

Monkey patches for HuggingFace Llama models. The default Llama patch handles
tree attention and `last_hidden_states`; the EAGLE3 patch additionally captures
low/mid/high base hidden states.

Patch selection is part of architecture. `tree_method="eagle3"` must use the
EAGLE3 patch dictionaries; other methods must keep the default H-dimensional
hidden-state path.

### `samd/tree_model/`

Draft-model integration layer.

* `token_recycle/` - Token Recycle draft model path.
* `eagle/` - EAGLE-1 integration.
* `eagle2/` - EAGLE-2 integration.
* `eagle3/` - EAGLE-3 integration. Uses `fc(3H -> H)`, a single
  `LlamaDecoderLayeremb`, independent `lm_head(H -> draft_vocab_size)`, and
  optional `d2t`/`t2d` vocabulary mapping.
* `fusion.py` - `TreeSpec(tokens, parents)` representation, EAGLE3 buffer
  conversion, and export to `tree_attn_mask`, `tree_position_ids`, and
  `tree_retrieve_indices`.

### `samd_sam_only/`

Optimized SAM-only path. It does not consume `tree_method` and should not be
used as a dumping ground for tree-model behavior.

### `evaluation/`

Benchmark runners and analysis scripts for Spec-Bench, MT-Bench, MedQA, and
MedQuAD-style data.

* `inference_samd.py` - main SAMD benchmark entry, including `tree_method`,
  `tree_fusion`, diagnosis trace, and `max_cache_len` flags.
* `inference_sam_only.py` - SAM-only benchmark entry.
* `eval_llama3.py` / `eval_vicuna.py` - question/turn loops and answer JSONL
  writing.
* `analyze_vmiss.py` - accept-length and V_miss markdown summary.
* `medqa_prep.py` / `medquad_prep.py` - dataset conversion to benchmark
  `question.jsonl`.

### `tools/static_sam/`

Offline Static SAM corpus and artifact pipeline. This layer should preserve data
provenance and keep corpora model-independent until artifact generation.

## Hard Boundaries

* `samd` currently supports Llama-family base models only.
* `SamdModel` and related tree paths support `batch_size == 1` only.
* Candidate generation may change candidate ordering/coverage, but verifier
  acceptance semantics remain the source of truth.
* `tree_method` and `tree_fusion` are independent dimensions and must stay
  independently configurable.
* Historical CodeStable docs are archive material; new durable architecture
  updates should land in this Trellis spec layer.
