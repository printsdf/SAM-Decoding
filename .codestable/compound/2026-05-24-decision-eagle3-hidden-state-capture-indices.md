---
doc_type: decision
category: constraint
date: 2026-05-24
slug: eagle3-hidden-state-capture-indices
status: active
area: samd/model_patch
tags: [eagle3, base-model-patch, hidden-states, training-inference-alignment]
---

## 背景

EAGLE3 训练时用 `../EAGLE/eagle/model/modeling_llama_kv.py:1137-1139` 在 base model 的特定层抓 hidden states 作为 draft 输入：

```python
for idx, decoder_layer in enumerate(self.layers):
    if idx == len(self.layers) - 3 or idx == len(self.layers) // 2 or idx == 2:
        all_hidden_states += (hidden_states,)
```

三个 index 是 `{2, N//2, N-3}`（low / mid / high），代表训练时模型学到的特征空间。

## 决定

samd 的 `llama_eagle3.py:llama_model_forward_eagle3` 在 patched LlamaModel.forward 内**硬编码同样三个 index** `{2, n_layers // 2, n_layers - 3}` 收集 hidden states（见 line 90）。**不允许**通过 config 让用户改。

## 理由

- **训练-推理对齐是 EAGLE3 性能的核心机制**：训练时学到的是这三层的特征空间，推理时换索引意味着用未训练的特征 → silent 输出退化（draft logits 看起来正常但与 base model expectations 错位 → accept rate 大跌）
- **用户 config 化引入 silent 错误风险**：用户传错索引（比如 `{0, 5, 10}`）代码不会 crash，但 accept_length 莫名其妙跌 —— 难以 debug
- 跟 EAGLE3 官方 base model fork 完全 1:1 对齐是最安全的选择

## 考虑过的替代方案

- **让 `Eagle3Config` 接受 `eagle_layers_to_capture` 字段**：被拒。理由如上 —— silent 错误风险
- **从 EAGLE3 权重的 config.json 读 capture indices**：被拒。官方 config.json 也没存这个字段（说明设计上就是硬编码）

## 后果

- 适用 base model 必须满足 `num_hidden_layers ≥ 6`（否则三个 index 会重合或越界 —— Llama-3.1-8B N=32 远超阈值 ✓）
- 未来 EAGLE 系列变种如果改了抓取策略（例如抓 4 层、抓不同 index），不能复用本 patch —— 需要新写 `llama_eagle4_patch_dict` 或类似
- 与 [[2026-05-24-decision-eagle3-base-patch-required]] 互为前提：那条决定保证 patch 被装上，本条决定保证装上后抓的层是训练对齐的

## 相关文档

- 代码：`samd/model_patch/llama_eagle3.py:88-98`
- 训练侧：`../EAGLE/eagle/model/modeling_llama_kv.py:1137-1139`
- 架构：`.codestable/architecture/ARCHITECTURE.md` 第 5 节
