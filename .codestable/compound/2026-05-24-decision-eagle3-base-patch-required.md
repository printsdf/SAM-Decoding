---
doc_type: decision
category: constraint
date: 2026-05-24
slug: eagle3-base-patch-required
status: active
area: samd/model_patch
tags: [eagle3, base-model-patch, tree-method, hidden-states]
---

## 背景

samd 通过 `samd_config.tree_method` 路由到不同 draft model 实现。EAGLE3 path 的 draft 模型期望 base model 输出 low/mid/high 三层 hidden state 拼接的 3H 张量（详见 [[2026-05-24-learning-eagle-variant-integration-checklist]] 维度 5），与 EAGLE / EAGLE2 / Token Recycle 只用 last layer hidden state 不同。

## 决定

当 `tree_method == "eagle3"` 时，`SamdModel.register_forward_patch` **必须**用 `eagle3_patch_dict` / `eagle3_attn_patch_dict`（不是默认的 `patch_dict` / `attn_patch_dict`）。该路由由 `samd/samd_model.py:64-77` if-else 显式分支保证。

## 理由

- 不装 eagle3 patch → `LlamaModel.forward` 不会在 idx ∈ {2, N//2, N-3} 收集 hidden states → `outputs.last_hidden_states` 是 H 维而非 3H 维 → Eagle3Model 的 `fc(3H→H)` 投影遇到 H 维输入，shape mismatch 立即 raise（**fail-fast** 优于 silent 退化）
- 如果通过 `if hidden_states.shape[-1] != inputs_embeds.shape[-1]:` 这个条件意外跳过 fc 投影（hidden 已是 H 维），draft model 会**用错误的 H 维特征跑 forward**，silent 输出 garbage —— 这才是真正的危险

## 考虑过的替代方案

- **合并 eagle3_patch_dict 到默认 patch_dict 全局生效**：被拒。其他 tree_method（eagle2 / token_recycle）的 forward 路径假设 outputs.last_hidden_states 是 H 维，强制 3H 会破坏向后兼容
- **在 Eagle3 集成层自己取多层 hidden state（hook 方案）**：被拒。生命周期管理复杂，且与 samd 既有 monkey patch 模式不一致

## 后果

- `tree_method` 切换时 `register_forward_patch` 必须重新跑（实际上每次构造 `SamdModel` 都会调一次，无需手动）
- 未来加新 tree_method 时，如果它也需要不同 base model patch（比如 EAGLE-4 抓 5 层），需要按本约束扩展 if-else 分支
- 与 [[2026-05-24-decision-eagle3-hidden-state-capture-indices]] 互为前提：本约束保证 patch 装上，那条约束保证装上后抓的层是对的

## 相关文档

- 代码：`samd/samd_model.py:64-77`、`samd/model_patch/llama_eagle3.py`
- 架构：`.codestable/architecture/ARCHITECTURE.md` 第 5 节
