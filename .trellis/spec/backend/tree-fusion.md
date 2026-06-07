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

## Scenario: Naive Fusion Uses EAGLE3 Logprobs

### 1. Scope / Trigger

This contract applies when `fusion_mode="naive"` ranks EAGLE3 candidates
against SAM candidates. It exists because depth is not an EAGLE3 confidence
signal; naive fusion must use the draft model's real per-token logprobs when
they are available.

### 2. Signatures

EAGLE3 model generation returns token-aligned logprobs at the raw model layer:

```python
draft_tokens, retrieve_indices, tree_mask, tree_position_ids, draft_logprobs = (
    Eagle3Model.topK_genrate(...)
)
```

The public tree-model wrapper must stay backward compatible:

```python
tokens, buffers = Eagle3.gen_draft(start_token)
tokens, buffers, logprobs = Eagle3.gen_draft(start_token, return_logprobs=True)
```

Naive fusion accepts logprobs through either the tree payload or an explicit
argument:

```python
parse_eagle_tree(eagle_tree, start_token, eagle_logprobs=None)
fuse_eagle_sam_naive(..., eagle_logprobs=None)
```

### 3. Contracts

`draft_logprobs` must be aligned one-to-one with final `draft_tokens` after
EAGLE3 top-score pruning and ordering. Index `0` is the root/start token and
uses a placeholder `0.0`; non-root indices use the local token logprob from the
parent expansion. Naive fusion uses `logprobs[i]` as the EAGLE node score for
token index `i`; if no logprobs are provided, it keeps the old depth-proxy
fallback for compatibility.

### 4. Validation & Error Matrix

* provided logprobs length differs from `draft_tokens` length -> `ValueError`.
* `eagle_tree["tokens"][0] != start_token` -> `ValueError`.
* `return_logprobs=False` -> return exactly `(tokens, buffers)` and avoid extra
  logprob list conversion on the default EAGLE3 path.

### 5. Good/Base/Bad Cases

* Good: `fusion_mode="naive"` requests `return_logprobs=True` and passes the
  returned list into `fuse_eagle_sam_naive`.
* Base: EAGLE3-only generation calls `gen_draft(start_token)` and receives the
  existing two-value return.
* Bad: scoring EAGLE nodes with `1.0 / (depth + 1)` when logprobs are present.

### 6. Tests Required

* Unit test that `parse_eagle_tree` maps non-root scores from aligned
  `logprobs[1:]`.
* Unit test that logprob length mismatch raises.
* Smoke test that `gen_candidates(..., fusion_mode="naive")` can call
  `gen_draft(..., return_logprobs=True)` and build verifier buffers.
* Compile changed Python files when the local environment lacks model
  dependencies.

### 7. Wrong vs Correct

Wrong:

```python
node.score = 1.0 / float(depth + 1)
```

Correct:

```python
node.score = float(eagle_logprobs[index])
```

## Implementation Rules

* Keep shared tree representation and export logic in `samd/tree_model/fusion.py`.
* Preserve `TreeSpec(tokens, parents)` as the internal representation for
  fusion operations.
* Export only standard `tree_attn_mask`, `tree_position_ids`, and
  `tree_retrieve_indices` to the verifier.
* Fusion fallback paths that return a single-source tree must still record
  source metadata for acceptance accounting. For `fusion_mode="naive"`, include
  fields such as `selected_eagle`, `selected_sam`, `selected_both`, and
  `node_sources` even when SAM is skipped.
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
