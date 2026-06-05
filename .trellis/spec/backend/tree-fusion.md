# Tree Fusion

> Contracts for combining SAM and EAGLE3 candidates.

## Configuration Model

Keep these two dimensions separate:

* `tree_method` selects the draft model family, for example `eagle3`.
* `tree_fusion` selects the candidate-combination strategy.

Current `tree_fusion` values:

* `none`
* `sam_sequence_graft`
* `sam_tree_union_prune`
* `eagle_prefix_sam_expand`

Fusion modes currently support only `tree_method="eagle3"`. Unsupported
combinations must raise, not silently fall back.

## Current Default Strategy

The current best SAM + EAGLE3 default is:

```text
tree_method="eagle3"
tree_fusion="sam_sequence_graft"
```

This Stage A strategy improved throughput over legacy SAMD+EAGLE3 by about
4.3% on both MT-Bench and MedQuAD and reduced V_miss in the recorded
experiments.

## Strategy Semantics

### `none`

Preserves legacy behavior: choose between SAM sequence and tree-model draft
according to existing SAM threshold/bias logic.

### `sam_sequence_graft`

Generate the full EAGLE3 tree first. When SAM finds a threshold-qualified raw
sequence, remove padding and graft that sequence as an additional branch. Export
the combined tree to standard tree buffers and let the target verifier accept or
reject candidates.

This strategy must not change greedy verifier semantics.

### `sam_tree_union_prune`

Broad Dynamic SAM tree union with budgeted pruning. The first m16/k4 experiment
was negative on throughput, mean accepted length, and V_miss. Do not make this
the default or increase its budget without redesigning the pruning objective.

### `eagle_prefix_sam_expand`

Conservative local expansion along EAGLE/Stage-A prefix anchors. Initial m4/k2
results were near-neutral or slightly positive, but not enough to replace
`sam_sequence_graft` as default. Use it as an ablation/follow-up variant unless
repeat experiments show a stable gain.

## Implementation Rules

* Keep shared tree representation and export logic in `samd/tree_model/fusion.py`.
* Preserve `TreeSpec(tokens, parents)` as the internal representation for
  fusion operations.
* Export only standard `tree_attn_mask`, `tree_position_ids`, and
  `tree_retrieve_indices` to the verifier.
* Do not prune EAGLE3 nodes from `sam_sequence_graft`; that strategy is for
  validating candidate complementarity, not learned candidate selection.
* Preserve Stage A leaves when implementing local prefix expansion variants.

## Evaluation Rules

Compare fusion variants against the recorded Stage A baseline using:

* mean accepted tokens;
* tokens per second and speedup;
* tree step count;
* V_miss rate from trace-ON phase2 runs.

Do not promote a fusion mode to default based on a tiny V_miss change that is
within run noise and has no stable throughput benefit.
