---
doc_type: learning
track: knowledge
date: 2026-05-24
slug: eagle-variant-integration-checklist
component: samd/tree_model
tags: [eagle, eagle2, eagle3, speculative-decoding, draft-model, integration-checklist]
---

## 1. 背景

samd 把多种 speculative-decoding draft model（Token Recycle / EAGLE / EAGLE2 / EAGLE3）作为可插拔 tree_method 接入。每个 EAGLE 变种之间架构差异不止"换权重"那么简单 —— 仅看 state_dict key 可能 silent 兼容但语义错位（EAGLE2/EAGLE3 权重里都有 `fc.weight`，但形状不同）。

从 EAGLE2 扩展到 EAGLE3 集成时识别出 5 个**必须逐项核对**的架构维度。后续若再接入 EAGLE 系列新变种（或类似设计的 lightweight draft model），直接拿这 5 项当 checklist。

## 2. 指导原则

集成 SafeAILab EAGLE 系列任意新变种前，按下面 5 项逐一核对原版 `cnets.py:Model`：

| # | 维度 | EAGLE / EAGLE2 | EAGLE3 |
|---|---|---|---|
| 1 | **`fc` 投影输入维** | `2*H`（embed + last hidden 拼接） | `3*H`（low/mid/high 三层 hidden 拼接） |
| 2 | **`lm_head` 来源** | 共享 base 模型 `lm.lm_head`（draft 不带自己的 lm_head） | **独立** `self.lm_head(H → draft_vocab_size)`，draft 训练时单独训 |
| 3 | **Vocab 大小** | 同 base `vocab_size`（draft 直接输出 base vocab id） | 可缩减 `draft_vocab_size < vocab_size`，需要 `d2t`/`t2d` 映射 buffer 把 draft id 翻回 base id；vocab 相等时官方 `del d2t,t2d` |
| 4 | **Decoder layer 结构** | 标准 `LlamaDecoderLayer`，q/k/v 输入维 `H` | **`LlamaDecoderLayeremb`**，内部 `torch.cat((input_emb, hidden), dim=-1)` 再过 attention，所以 q/k/v 输入维 `2*H` |
| 5 | **Base model hidden state 抓取** | 只取最后一层 hidden state | 抓三层 `idx ∈ {2, N//2, N-3}` 拼接成 3H，在 base model fork 的 `LlamaModel.forward` 内硬编码 |

## 3. 为什么重要

- **`fc` 形状不同 → `load_state_dict` 立即 fail**（这是最显式信号；如果 weight key 都对得上 silent 加载完成，看 forward 行为才发现 silent 数值错）
- **`lm_head` 来源不同**：用错会导致 draft 输出在 base 模型不认识的 token id 上（独立 head 输出 32k token，base 期望 128k token id）
- **Vocab 映射缺失 / 错位**：`d2t` 全零 buffer 加载完仍然能跑，但生成的 token id 没经过偏移翻译，base model 验证全 fail
- **Decoder layer 类型不同**：复用 EAGLE2 的 `LlamaDecoderLayer` 给 EAGLE3 用，attention 的 q/k/v 投影维度不对，权重加载 size mismatch
- **抓取层错位**：用 EAGLE3 的 fc(3H→H) 配上 base model 的 last-layer-only hidden（H 维），forward 会跳过 fc 投影（`if hidden_states.shape[-1] != inputs_embeds.shape[-1]:` 这条 if 永远 false），结果维度不对但不报错，silent 退化为 garbage 输出

## 4. 何时适用

- 集成 SafeAILab/EAGLE 系列新变种到 samd
- 接入类似设计的 lightweight speculative-decoding draft model（Medusa / Hydra / 同类 head-only 架构）
- 跨 EAGLE 版本升级（如 EAGLE-2 → EAGLE-3 → 未来 EAGLE-4）

不适用：完全独立的 draft 模型（自己有 lm_head 又有完整 transformer，且不依赖 base model 任何 hidden state） —— 那种是"独立小 LLM"路径，不在本 checklist 覆盖范围内。

## 5. 示例

本仓库 EAGLE3 集成实操：
- 维度 1：`samd/tree_model/eagle3/eagle3_model.py:Eagle3Model.__init__` 的 `self.fc = nn.Linear(config.hidden_size * 3, ...)`
- 维度 2：`Eagle3Model.__init__` 的 `self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)`
- 维度 3：注册 `d2t`/`t2d` buffer；`topK_genrate` 内 `if vocab_size == draft_vocab_size:` 分支保留官方两种模式
- 维度 4：`samd/tree_model/eagle3/eagle3_utils.py:LlamaAttention` 的 `q/k/v_proj = nn.Linear(self.hidden_size * 2, ...)` + `LlamaDecoderLayeremb.forward` 内 `torch.cat((input_emb, hidden_states), dim=-1)`
- 维度 5：`samd/model_patch/llama_eagle3.py:llama_model_forward_eagle3` 的 `eagle3_targets = {2, n_layers // 2, n_layers - 3}` + 收集后 concat 成 3H 暴露给上游

前置探索文档：`.codestable/compound/2026-05-23-explore-eagle3-compatibility.md`（详细对比 EAGLE3 与 EAGLE2 的 8 处差异）。

后续如果接入 EAGLE-4 或其他变种，按这 5 维度对照 `cnets.py` 新版本写差异表，逐项决定哪些复用 EAGLE3 实现、哪些重新写。
