---
doc_type: learning
track: knowledge
date: 2026-05-24
slug: external-review-blocker-triage
component: workflow
tags: [code-review, codex, gemini, ai-reviewer, debugging]
---

## 1. 背景

CodeStable 工作流的 `cs-feat-impl` 阶段允许引入外部 AI reviewer（codex / gemini 等 MCP 工具）做 patch 审查。一个 reviewer 标"Blocker"时 —— 特别是关于 shape mismatch / length mismatch / state machine 这类静态分析能看出来的"硬错误"—— 我们的本能反应是马上按它指的方向修。

但本次 EAGLE3 集成踩过的一个坑表明：**Reviewer 标的 Blocker 不一定是真 bug，可能只是被 reviewer 缺少的调用约定信息误判**。盲目按 reviewer 改，可能修对一个 symptom 引入另一个 problem。

## 2. 指导原则

外部 reviewer 给出"Blocker / Critical bug"反馈时，**在动手改之前**做三步验证：

1. **从代码静态 trace shape / 类型**：在 reviewer 指出的代码点，画一遍**实际调用路径下**的张量 shape / 状态值。reviewer 通常看代码片段不一定有完整调用上下文。

2. **对比原版项目（如果代码移植自第三方）的同点行为**：移植代码出 bug 时第一反应应该是"原版项目也这样吗？" 如果原版也有同样代码且能跑通，那 99% 是**调用约定差异**而不是"代码本身 bug"。

3. **找原项目的实际调用方**：如 EAGLE 项目里 `utils.py:update_inference_inputs` 是 `topK_genrate` 的真实调用方 —— 看它传入参数的 shape 和 reviewer 假设的 shape 是不是一致。一致 → reviewer 没拿到完整上下文，他指的 Blocker 在原版语义下不存在。不一致 → 才考虑 reviewer 是否对。

只有三步都过，才动手改。如果改了，**还要验证修复后的行为是否真的恢复原版预期**（不止"length 对上"，还要看性能 / 输出质量这些 silent 指标）。

## 3. 为什么重要

外部 reviewer 报 false positive Blocker 的代价：

- **不修 → 留 bug**（reviewer 对的话）
- **修了 → 引入另一个 silent regression**（reviewer 错的话）

这次 EAGLE3 集成实际发生的：

1. Gemini 标 Blocker A："`Eagle3Model.topK_genrate` `if stable_kv: input_ids[:, kv_len:]` 分支下，`hidden_states` 没切片，与 sliced input_ids 长度不匹配，下游 `LlamaDecoderLayeremb.forward` 的 `torch.cat((input_emb, hidden_states), dim=-1)` 会 crash"
2. **盲目按 Gemini 修**：删掉 `if-stable_kv` 增量分支，改为永远全长 forward。length mismatch 不再发生 → 测试跑通 ✓
3. **后续 cs-feat-accept 阶段才发现**：accept_length 平均 2.35 远低于 EAGLE3 项目 4-6+。深挖原因 — 修复破坏了 EAGLE3 stable_kv 增量优化（**真正的根因是 samd 累积约定 vs EAGLE3 原版增量约定不匹配**，不是代码 bug）
4. **正确解法**：维护 `cumulative_tokens` 持久累积 + `pending_hidden_states` 消费即清 两套状态，恢复 cnets `if-stable_kv` 分支。修复 length mismatch 的同时保留增量优化

如果当时按"对比原版"原则（cnets.py 同点也是这样写、EAGLE 项目跑通了 → 一定是调用约定差异），可以直接走向正确解法，避免这次绕路。

## 4. 何时适用

- 使用 codex / gemini / 任何 AI reviewer 跑 patch review 时
- 移植 / 集成第三方代码到本项目时（移植代码的 review 反馈尤其要警惕调用约定差异）
- 收到"shape mismatch / length mismatch / type mismatch"这类静态分析类 Blocker 时

不适用：
- Reviewer 指出的是逻辑错误（off-by-one、错误的算法、wrong condition）—— 这类不涉及调用约定，直接验证逻辑就行
- 新写的代码（不是移植的）—— 没有"原版"可对比，按常规 debug 流程

## 5. 示例

**Anti-pattern**（这次踩过的）：

```
Gemini: "stable_kv 分支下 hidden_states 没切片，会 crash"
→ 我：好，删 stable_kv 分支，每次全长 forward
→ 测试跑通 ✓ — 但 accept_length 偏低
→ cs-feat-accept 阶段才发现性能损失，回头重做
```

**Correct pattern**（应该做的）：

```
Gemini: "stable_kv 分支下 hidden_states 没切片，会 crash"
→ Step 1: trace 实际 shape — pending_hidden 长 N，input_ids[:, kv_len:] 长 N - kv_len，确实不匹配
→ Step 2: 看 cnets.py 同点 — 一模一样代码，EAGLE 项目能跑
→ Step 3: 看 EAGLE 项目的 topK_genrate 调用点 — utils.py:454-468 传入的 hidden_states 是 accept_length+1 长（delta，不是累积全长）
→ 结论：cnets.py 调用约定是 "hidden_states 是 delta"，samd 累积全长破坏了约定
→ 真正解法：让 Eagle3 维护两套状态桥接累积 vs 增量约定，不是改 cnets
```

相关详细记录：[[2026-05-24-learning-samd-eagle3-incremental-state-machine]]

## 关联工具

- codex MCP（read-only review 模式）：`mcp__codex__codex` with `sandbox="read-only"`
- gemini MCP：`mcp__gemini__gemini`
- 推荐 prompt 风格：让 reviewer 输出"Severity: blocker/major/minor/nit + File:line + Issue + Suggested fix"结构化反馈，便于逐条 triage
