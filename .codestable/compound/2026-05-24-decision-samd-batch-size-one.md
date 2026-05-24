---
doc_type: decision
category: constraint
date: 2026-05-24
slug: samd-batch-size-one
status: active
area: samd
tags: [batch-size, single-sequence, speculative-decoding, scope]
---

## 背景

speculative decoding 通过 tree-attention 在单个 sequence 内部并行验证多个 candidate token。samd 的核心数据结构（tree_attn_mask / tree_position_ids / tree_retrieve_indices）和 StaticCache 选 indices 路径，都按 batch_size=1 假设设计。

`samd/samd_model.py:240` 有显式断言：

```python
assert input_ids.shape[0] == 1, "Only support batch_size == 1"
```

## 决定

samd 仅支持 **`batch_size == 1`**。多 sequence 并发推理需要外层调度（多进程 / 多 GPU / 排队），不在 samd 内实现。

## 理由

- speculative decoding 的 acceptance check 是按 sequence 独立计算的（不同 sequence 的 accept_length 不同），batched tree attention 在工程实现上需要 padding / per-sequence mask，复杂度跳变
- 现有 tree_attn_mask 是 `[1, 1, T, T]` 形状，扩展到 `[B, 1, T, T]` 要改 candidate generation / SAM lookup / KV cache slice 多处，工作量大
- README 列出的所有 benchmark（Spec-Bench）都按 batch_size=1 评测 — 这是 speculative decoding 领域的事实标准
- 研究项目优先聚焦"单序列 acceleration 比"

## 考虑过的替代方案

- **支持 batched 推理**：被拒。需要重新设计 cache / mask / lookup 几乎所有数据结构；脱离当前 research scope。如果未来上 production 应用，建议另开 feature 而不是在当前代码基上扩展
- **External batching（多进程）**：可行的替代，但不算 samd 自身的功能 — 用户自己写 launcher 多开几个 samd 进程即可

## 后果

- `SamdModel.generate` 入参 `input_ids.shape[0]` 必须为 1，否则在 assert 处立即 fail
- `samd/tree_model/eagle3/` 的所有 hidden state shape 假设 `[B=1, S, ...]`，未来如果突破该约束需要全面重审 tree_model 实现
- evaluation 入口 `evaluation/inference_samd.py` 已经按单 sequence 跑 benchmark，无需改

## 相关文档

- 代码：`samd/samd_model.py:240`（显式 assert）
- 架构：`.codestable/architecture/ARCHITECTURE.md` 第 5 节
