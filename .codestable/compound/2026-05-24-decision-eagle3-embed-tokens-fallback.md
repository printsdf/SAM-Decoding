---
doc_type: decision
category: constraint
date: 2026-05-24
slug: eagle3-embed-tokens-fallback
status: active
area: samd/tree_model/eagle3
tags: [eagle3, weight-loading, embed-tokens, base-model-coupling]
---

## 背景

EAGLE3 官方权重发布（如 `/root/Models/EAGLE3-LLaMA3.1-Instruct-8B`）**不包含** `embed_tokens.weight` —— `Eagle3Model.embed_tokens` 在训练时用 `load_emb=True` 从 base model 路径直接加载（见 `../EAGLE/eagle/model/cnets.py:488-519`），训练后保存的 state_dict 不再 dump 它（draft 共享 base 的 embedding，`requires_grad=False`）。

samd 集成层不直接接受 `base_model_path`，且我们移植 `Eagle3Model` 时去掉了 `load_emb` 参数（推理用不上 hf_hub 兜底）。

## 决定

`Eagle3.__init__` 在 `load_weight(tree_model_path)` 之后、`init_tree()` 之前，**必须**从已加载的 base model 的 `lm.model.embed_tokens.weight.data` 复制到 `self.model.embed_tokens.weight.data`。当 base 和 draft 的 vocab_size 不一致时跳过复制并打印警告（draft 会以零张量推理，输出无意义）。

`Eagle3Model.load_weight` 的 strict 检查中，`embed_tokens.weight` **始终**作为 allowed missing key（由 Eagle3 集成层兜底）。

## 理由

- 不复制 → `Eagle3Model.embed_tokens.weight` 留在零初始化状态 → `self.embed_tokens(input_ids)` 出零张量 → fc 投影后送 attention 全部退化 → draft 永远预测错的 token
- 不允许 missing → strict 加载报错 → 用户无法用官方 EAGLE3 权重，必须自己合 base embedding 进去（不现实）
- 用 base 的 embedding 是 EAGLE3 的设计假设（draft 与 base 共享 vocab embedding space），不是 workaround

## 考虑过的替代方案

- **要求权重包含 `embed_tokens.weight`**：被拒。这违背 EAGLE3 训练侧设计，且 SafeAILab 官方发布的所有 EAGLE3 权重都不带
- **加 `load_emb` 参数从 base_model_path 直接读 safetensors**：被拒。samd 集成层已经有 base model 实例（`lm.model.embed_tokens.weight`），从内存复制比从磁盘读更直接 + 省一次 I/O；且能利用已经移到 device 上的 tensor

## 后果

- Eagle3 与 base model 实例**强耦合** — 不能在没有 base lm 的情况下构造 Eagle3（构造签名也强制要求 `lm: LlamaForCausalLM` 参数）
- base 和 draft 的 vocab 必须一致才能正确工作。Llama-3.1（vocab=128256）+ EAGLE3-LLaMA3.1（也 128256）一致 ✓；如果出现 base/draft vocab 不一致的"野生"组合，会打印警告但不阻塞构造（用户应该看到警告并停下来）

## 相关文档

- 代码：`samd/tree_model/eagle3/eagle3.py:30-44`、`samd/tree_model/eagle3/eagle3_model.py:load_weight`
- 架构：`.codestable/architecture/ARCHITECTURE.md` 第 5 节
- 实践细节：[[2026-05-24-learning-eagle-variant-integration-checklist]]
