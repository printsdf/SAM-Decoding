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
tokens, buffers, logprobs, raw_logits = Eagle3.gen_draft(
    start_token, return_logprobs=True, return_raw_logits=True
)
```

The 4-tuple is the offline-calibration path used by the fusion profiler: each
`raw_logits[i]` is the parent's top-1/top-2 raw logits `(z1, z2)` aligned with
`draft_tokens[i]`. The root/start token (index 0) has no parent expansion and
carries `(None, None)`. Online callers must continue to use the 2-tuple or
3-tuple; the 4-tuple is profiler-gated and has negligible cost (gather 2
values per parent).

Naive fusion accepts logprobs and raw logits through either the tree payload
or an explicit argument:

```python
parse_eagle_tree(eagle_tree, start_token, eagle_logprobs=None, eagle_raw_logits=None)
fuse_eagle_sam_naive(..., eagle_logprobs=None, eagle_raw_logits=None)
```

### 3. Contracts

`draft_logprobs` must be aligned one-to-one with final `draft_tokens` after
EAGLE3 top-score pruning and ordering. Index `0` is the root/start token and
uses a placeholder `0.0`; non-root indices use the local token logprob from the
parent expansion. Naive fusion uses `logprobs[i]` as the EAGLE node score for
token index `i`; if no logprobs are provided, it keeps the old depth-proxy
fallback for compatibility.

`raw_logits[i] = (z1, z2)` records the parent's raw top-1/top-2 logits (the
parent that expanded `draft_tokens[i]`). MARS-style ratio analysis derives
`z2 / (z1 + 1e-10)` from these; the node's own raw logit is not stored. Old
traces without `raw_logits` still load; `parse_eagle_tree` sets
`parent_top1_logit`/`parent_top2_logit` to `None` in that case.

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

## Scenario: Fixed-Budget Drafter-MARS Repair

> **Experimental status (2026-07-17): negative ablation.** Keep
> `drafter_mars_tree_budget=None` as the default. On HumanEval, both post-graft
> pruning to 60 nodes and preallocating EAGLE slots to SAM lost MAT and speedup
> versus append-only repair. Do not present a fixed total-node budget as a
> recommended optimization without a new branch-utility selector and fresh
> same-boot evidence.

### 1. Scope / Trigger

This contract applies when `fusion_mode="drafter_mars"` appends a SAM repair or
leaf extension and `drafter_mars_tree_budget` is set. It exists because higher
MAT does not translate into throughput when the appended nodes increase the
target verifier's per-step cost.

### 2. Signatures

```text
--drafter_mars_tree_budget <positive integer>

SamdConfig.drafter_mars_tree_budget: Optional[int]

prune_eagle_leaves_to_budget(
    tree: TreeSpec,
    *,
    eagle_node_count: int,
    eagle_logprobs: Sequence[float],
    max_total_nodes: int,
    protected_indices: Optional[Iterable[int]],
) -> Tuple[TreeSpec, Dict[str, Any]]
```

### 3. Contracts

* The budget is root-inclusive. With stock `eagle3_total_token=60`, budget 60
  keeps a repaired verifier tree at the stock EAGLE3 size.
* `None` preserves the append-only behavior exactly.
* Direct pruning candidates are original EAGLE nodes only. Newly appended SAM
  nodes, their ancestor closure, root, and the EAGLE greedy top path are
  protected.
* Pruning is leaf-only and iterative, so every surviving node retains a valid
  parent. Surviving nodes are remapped to contiguous parent-before-child
  indices before `TreeSpec.to_buffers()`.
* Candidate priority is ascending mean path logprob: the least likely EAGLE
  leaf is removed first. Ties prefer deeper and later-indexed leaves.
* The same helper applies to single graft, multi-graft, SAM subtree repair, and
  greedy-leaf extension.
* Runtime metadata includes `tree_budget`, `pre_prune_nodes`,
  `pruned_eagle_nodes`, and root-inclusive `final_total_nodes`; the legacy
  `final_nodes` field remains non-root node count.

### 4. Validation & Error Matrix

* budget is `None` -> pruning disabled.
* budget is bool, non-integer, or `< 1` -> `ValueError` during config/function
  validation.
* EAGLE logprob length differs from original EAGLE node count -> `ValueError`.
* protected/SAM paths alone exceed the requested budget -> `ValueError`; never
  silently prune SAM nodes or exceed the verifier budget.

### 5. Good/Base/Bad Cases

* Good (mechanical invariant test only): stock 60-node EAGLE tree + eight SAM
  nodes, budget 60 -> prune eight low-value EAGLE leaves and verify exactly 60
  total nodes. This demonstrates correctness, not expected speedup.
* Base: `drafter_mars_tree_budget=None` -> retain the current EAGLE+SAM tree.
* Bad: truncate the fused token array to 60; this can orphan descendants,
  remove the greedy path, or leave invalid retrieve indices.
* Bad: reduce `eagle3_total_token` merely to reserve SAM slots and claim an
  equal-node comparison is cost-equivalent. The verifier is sensitive to tree
  shape and candidate coverage, not only node count.

### 6. Tests Required

* Synthetic tree: assert exact final budget, valid remapped parents, SAM path
  retention, greedy-path retention, and removal of the lowest-scored EAGLE
  leaves.
* No-op tree: assert object/structure unchanged when already within budget.
* Impossible budget: assert a hard error.
* Config: accept 60; reject zero and bool.
* Server batch: compare append-only r8/e16, budget-60 r8/e16, and budget-60
  r8/K2/e16 using MAT, total time, tok/s, average final nodes, and verify time.

### 7. Wrong vs Correct

Wrong:

```python
fused_tokens = fused_tokens[:60]
```

Correct:

```python
fused_tree, stats = prune_eagle_leaves_to_budget(
    fused_tree,
    eagle_node_count=len(eagle_tokens),
    eagle_logprobs=eagle_logprobs,
    max_total_nodes=60,
    protected_indices=greedy_path,
)
```
