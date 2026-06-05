# Runtime Constraints

> Dependency, environment, tokenizer, and cache constraints.

## Dependencies

Any `samd` import or test path depends on these packages:

* `torch`
* `transformers` 4.46.x
* `safetensors`

Missing packages can fail immediately at import time. Do not run or report SAMD
tests as meaningful unless these dependencies are installed.

## Transformers Version Pin

SAMD monkey patches copy and modify HuggingFace Llama internals from
`transformers` 4.46.x. The project must stay pinned to 4.46.x unless the patch
files are deliberately rewritten and reviewed.

Why this matters:

* 4.50+ moves `StaticCache` and changes causal-mask APIs.
* 4.50+ changes `LlamaModel.forward` parameters and position-embedding flow.
* Both default and EAGLE3 Llama patches are coupled to the 4.46.x shape.

## Batch Size

`SamdModel` supports only `batch_size == 1`. The tree attention masks, retrieve
indices, StaticCache slicing, SAM lookup, and acceptance accounting all rely on
single-sequence assumptions. Multi-sequence throughput should be handled by an
outer scheduler, not by locally lifting the assert.

## Cache Length

For Llama-3.1 and other long-context base models, pass an explicit
`--max_cache_len <= 4096` when running `evaluation/inference_samd.py` or
`evaluation/inference_sam_only.py` on 24 GiB GPUs. If omitted, these runners may
default to `model.lm.config.max_position_embeddings`, which can allocate a huge
KV cache.

Tests and library examples generally use `max_cache_len=2048`.

## Tokenizer Padding

Llama-3-family tokenizers may have `pad_token=None`. Before using
`padding=True`, preserve this fallback:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

Do not remove this from tests or tokenizer setup paths unless the tokenizer is
known to define a pad token.

## Supported Backbones

The active `samd/model_patch/` code supports Llama-style models. Qwen2/Qwen3,
Mixtral, and other backbones are not supported by the current patch layer unless
a task explicitly adds and validates a new patch family.

## Static SAM Inputs

Static SAM artifacts must be built from trusted offline corpora with documented
source and tokenizer provenance. Do not build Static SAM from evaluation
benchmark data.
