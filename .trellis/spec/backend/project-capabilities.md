# Project Capabilities

> Current capability intent migrated from CodeStable requirements.

## Current Capabilities

### Evaluation V_miss Diagnosis

Users need to compare speculative decoding strategies on their own datasets and
understand whether draft vocabulary coverage explains speedup loss.

The system provides:

* optional per-step diagnosis trace;
* MedQA/MedQuAD-style data preparation;
* three/four group benchmark scripts;
* `evaluation/analyze_vmiss.py` summaries for accept length and V_miss.

Boundaries:

* This capability does not evaluate answer correctness.
* V_miss is meaningful for EAGLE3 paths; other tree methods may leave fields
  empty or N/A.
* Trace is default-off and should not affect normal inference performance.

### SAM + EAGLE3 Tree Fusion

Users need SAM suffix-match candidates and EAGLE3 draft-tree candidates to be
verified in the same decode step instead of choosing only one source.

The system provides:

* `tree_fusion` configuration separate from `tree_method`;
* Stage A `sam_sequence_graft` fusion for adding a raw SAM branch to the full
  EAGLE3 tree;
* experimental Stage B/B2 variants for broader tree union and local prefix
  expansion.

Boundaries:

* Fusion currently supports only EAGLE3.
* Candidate fusion must not change the verifier's final acceptance semantics.
* Static SAM data sourcing is a separate capability.

## Draft Capability

### Static SAM Offline Corpus

Users need trustworthy vertical-domain Static SAM corpora that do not leak
benchmark data into acceleration experiments.

The intended system shape:

* normalize local medical/finance/instruction QA data into a model-independent
  offline corpus;
* record source, domain, license/permission, and evaluation exclusion rationale;
* generate model-style responses only after corpus normalization;
* build tokenizer/model-specific Static SAM artifacts as the final step.

Boundaries:

* Do not auto-ingest every public dataset.
* Do not include common benchmark/evaluation datasets in Static SAM.
* Do not change the SAM-Decoding runtime algorithm as part of corpus work.

## Historical Notes

Detailed CodeStable requirement and feature records were migrated into Trellis
summaries and the old `.codestable/` directory was removed. Treat this file as
the active Trellis capability summary; use the migration inventory under
`.trellis/tasks/06-05-codestable-to-trellis/research/` for traceability.
