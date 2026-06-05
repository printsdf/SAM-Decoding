# EAGLE3 Integration

> Contracts for `tree_method="eagle3"`.

## Architecture Contract

EAGLE3 is not a drop-in EAGLE2 weight swap. Before changing or adding an EAGLE
variant, compare the original model along these dimensions:

| Dimension | EAGLE/EAGLE2 | EAGLE3 |
| --- | --- | --- |
| `fc` input | `2H` from embedding plus last hidden | `3H` from low/mid/high base hidden states |
| `lm_head` | Shared base model head | Independent `lm_head(H -> draft_vocab_size)` |
| Vocabulary | Base vocab IDs | Optional reduced draft vocab with `d2t`/`t2d` mapping |
| Decoder layer | Standard Llama decoder layer | `LlamaDecoderLayeremb` using embedding plus hidden input |
| Base hidden states | Last hidden only | Fixed layers `{2, N//2, N-3}` concatenated to `3H` |

## Required Base Patch

When `tree_method == "eagle3"`, `SamdModel.register_forward_patch` must install
`eagle3_patch_dict` and `eagle3_attn_patch_dict`. The default patch returns
H-dimensional hidden states and is invalid for EAGLE3's `fc(3H -> H)` input.

Do not make the EAGLE3 patch global. EAGLE/EAGLE2/Token Recycle paths expect
the default H-dimensional hidden-state contract.

## Hidden-State Capture Indices

The EAGLE3 base patch must capture the same three layers used by EAGLE3
training:

```text
{2, n_layers // 2, n_layers - 3}
```

Do not expose these indices as user config. Wrong indices can produce normal
looking logits with poor acceptance behavior, which is harder to debug than a
fail-fast error.

The base model must have enough layers for these indices to be distinct and
valid. Llama-3.1-8B with 32 layers is safe.

## Official Weight Loading

Official EAGLE3 checkpoints omit `embed_tokens.weight` because the draft model
shares the base embedding space. The integration must:

* allow `embed_tokens.weight` as a missing key during draft weight load;
* copy `lm.model.embed_tokens.weight.data` into the draft embedding after load
  when base and draft shapes match;
* warn if base and draft vocab shapes differ.

A base/draft vocab mismatch makes draft output meaningless even if construction
continues.

## Incremental State Machine

EAGLE3's original `cnets.py` path relies on `stable_kv` incremental forward.
The SAMD integration bridges that with two states:

* `cumulative_tokens` - full token history for input IDs; cleared only by
  `reset()`.
* `pending_hidden_states` - hidden-state delta since the last `gen_draft`;
  consumed and cleared by `gen_draft`.

`gen_draft` passes full `input_ids` history plus delta hidden states. Do not
replace this with full hidden-state accumulation; that can pass shape checks
while destroying EAGLE3's performance mechanism.

## Vocabulary Mapping

EAGLE3 can use `draft_vocab_size < vocab_size`. Preserve `d2t` and `t2d`
mapping behavior so draft token IDs are translated back to base token IDs. When
the sizes are equal, preserve the official simplification path.

When an official checkpoint omits `d2t`/`t2d` because `draft_vocab_size ==
vocab_size`, default `t2d` must be all `True`, not all `False`. This preserves
identity reachability for diagnosis traces and prevents later sidecar code from
misclassifying the whole vocabulary as V_miss.

## Tail Sidecar Tree-Budget Contract

### 1. Scope / Trigger

This contract applies when wiring EAGLE3 tail sidecar inference into SAMD or
running official-tail comparisons. It was added because the official tail eval
uses a different EAGLE3 tree budget than the old cnets defaults, and mismatched
budgets can look like a tail quality bug.

### 2. Signatures

`SamdConfig` must expose these EAGLE3-only fields:

```python
eagle3_tail_path: Optional[str] = None
eagle3_tail_type: Literal["auto", "plain", "tucker"] = "auto"
eagle3_total_token: int = 60
eagle3_depth: int = 7
eagle3_top_k: int = 10
```

`evaluation/inference_samd.py`, `samd/inference/cli.py`, and EAGLE3 smoke tests
must accept matching flags:

```text
--eagle3_tail_path
--eagle3_tail_type
--eagle3_total_token
--eagle3_depth
--eagle3_top_k
```

### 3. Contracts

The official tail comparison budget is:

```text
total_token=60, depth=7, top_k=10
```

Shell runners should read:

```text
EAGLE3_TOTAL_TOKEN=60
EAGLE3_DEPTH=7
EAGLE3_TOP_K=10
```

and pass those values through to `evaluation.inference_samd`. These are separate
from SAM fusion budgets such as `sam_tree_top_k` and `sam_prefix_top_k`.

### 4. Validation & Error Matrix

* `eagle3_tail_path == ""` -> normalize to `None`.
* `eagle3_tail_path is not None` with `tree_method != "eagle3"` -> `ValueError`.
* `eagle3_tail_type` outside `auto/plain/tucker` -> `ValueError`.
* non-positive or boolean `eagle3_total_token`, `eagle3_depth`, or
  `eagle3_top_k` -> `ValueError`.

### 5. Good/Base/Bad Cases

* Good: official-tail eval passes `60/7/10` and sees the EAGLE3 load log print
  those values.
* Base: no tail path uses the same tree-budget fields for pure EAGLE3 baseline.
* Bad: using old `63/5/8` cnets defaults while comparing against official tail
  numbers; this changes the candidate tree and can depress accept length.

### 6. Tests Required

* Syntax-check modified runners with `bash -n`.
* Compile changed Python entry points with `python -m compileall`.
* In a model environment, run baseline and tail with the same `60/7/10` budget
  before attributing accept-length deltas to tail quality.

### 7. Wrong vs Correct

Wrong:

```bash
python -m evaluation.inference_samd --tree_method eagle3 --eagle3_tail_path tail.pt
```

Correct for official-tail comparison:

```bash
python -m evaluation.inference_samd \
  --tree_method eagle3 \
  --eagle3_tail_path tail.pt \
  --eagle3_total_token 60 \
  --eagle3_depth 7 \
  --eagle3_top_k 10
```

## Review Checklist

* Does `tree_method="eagle3"` still select the EAGLE3 patch?
* Are hidden states still `3H` at the EAGLE3 draft input?
* Are capture indices still `{2, N//2, N-3}`?
* Is `embed_tokens.weight` missing handled as expected?
* Does `stable_kv` incremental behavior remain intact?
* Are `d2t`/`t2d` mappings preserved for reduced draft vocab checkpoints?
* Are EAGLE3 tail/baseline comparisons using the same `total_token/depth/top_k`
  budget, preferably the official `60/7/10` setting?
