---
doc_type: learning
track: pitfall
date: 2026-05-24
slug: samd-eagle3-incremental-state-machine
component: samd/tree_model/eagle3
severity: high
tags: [samd, eagle3, kv-cache, draft-model, calling-convention]
---

## 1. 问题

移植 `../EAGLE/eagle/model/cnets.py:Model` 到 samd 作为新 `tree_method="eagle3"` 时，按 samd 现有 `Eagle2` baseline 风格让 `Eagle3.update` 累积全长 hidden_states，`gen_draft` 把累积全长丢给 `topK_genrate`。这破坏了 cnets 设计上的 `stable_kv` 增量 forward 优化 —— 每次 gen_draft 都重算全部累积长度的 KV，浪费大量计算且产生 KV 重复 cache 隐患。

## 2. 症状

- `accept_length` 平均 ~2.35（EAGLE3 论文 / 官方 typical 4-6+）
- Wall time 不降反升的隐患（在 single deterministic prompt + Llama-3 instruct 上因为 argmax 稳定，wall time 差异只在 ~5% 量级，肉眼难察觉）
- 长 generation 下 GPU 显存会异常增长（每次 gen_draft 把累积 KV cat 一份新的进去）

## 3. 没用的做法

- **直接对齐 `Eagle2` baseline 累积约定**（删掉 cnets 的 `if hasattr(stable_kv): input_ids[:, kv_len:]` 分支，每次全长 forward）—— 这是外部 reviewer (Gemini) 指出 Blocker A "stable_kv 分支下 hidden_states 没切片导致 length mismatch" 后的第一反应修复。逻辑上修对了 length mismatch（forward 内部不会 crash），但**完全丢失 EAGLE3 性能机制**。
- **改用 forward hook 收集**（forward-scoped hook 在 patched `LlamaForCausalLM.forward` 内部 attach/detach）—— 与 samd 现有 monkey patch 风格不一致；design D2 拒绝过
- **`output_hidden_states=True` + 在 Eagle3 内取 idx**—— LlamaModel 内部会 allocate 全 N 层 hidden states，浪费 `(N-3) × seq × H` 显存

## 4. 解法

让 `Eagle3` 维护**两套状态**桥接 samd 累积框架与 cnets 增量约定：

- `self.cumulative_tokens` — 持久累积全部 input_ids 历史；**只在 `reset()` 清**，不在 `gen_draft` 清
- `self.pending_hidden_states` — 自上次 `gen_draft` 以来未消费的 hidden_states delta；**`gen_draft` 末尾清**

`Eagle3.gen_draft` 调 `topK_genrate(pending_hidden[None], (cumulative_tokens + start_token)[None], head)` —— hidden_states 是 delta（长度 ≈ accept_length），input_ids 是累积全长。

`Eagle3Model.topK_genrate` 恢复 cnets.py 原版 `if hasattr(self, "stable_kv") and self.stable_kv is not None: input_ids[:, kv_len:]` 增量分支 + `len_posi = input_ids.shape[1]`（相对长度作为 position 基准，因为剥过第一个后正好对应下一个待生成位置）。

代码落点：
- `samd/tree_model/eagle3/eagle3.py:Eagle3.{__init__, reset, update, gen_draft}`
- `samd/tree_model/eagle3/eagle3_model.py:Eagle3Model.topK_genrate`

## 5. 为什么有效

新约定下，每次 `gen_draft`：

- `pending_hidden_states` 长度 = 这轮 accept_length（小，~3-6）
- `input_ids_full[:, 1:]` 长度 = K_prompt + Σaccept（大）
- `input_ids[:, kv_len:]` slice 长度 = `Σaccept_this_round`（小），与 `pending_hidden_states` 长度匹配 ✓
- 主 forward 只算 delta 长度的 KV，过去的 KV 通过 `stable_kv` 复用 —— **复刻 EAGLE3 原版 `../EAGLE/eagle/model/utils.py:update_inference_inputs:454-468` 的 `accept_hidden_state_new` 调用约定**

`stable_kv` 跨 gen_draft 复用，KV cache 自然累积长度 = `K_prompt + Σall_accepts`，无重复 cache。

Invariant：
- prefill 后第一次 gen_draft：`stable_kv=None` → 走 else 分支全长 forward（等价于无优化）
- accept(a) 后第 N 次 gen_draft：`stable_kv` 长 `K_prompt + Σ_{i<N} a_i`，本次 forward 只算 `a_{N-1} + 1` 长

## 6. 预防

移植任何 `cnets.py` 风格的"轻量 draft model"项目时，**先确认原项目的调用约定**：

1. 看原项目 `topK_genrate` 入口处 `if hasattr(self, "stable_kv")` 分支是否存在 → 存在意味着设计上依赖增量
2. 找原项目实际调用 `topK_genrate` 的位置（如 `utils.py:update_inference_inputs`、`ea_model.py:eagenerate` 等），看 `hidden_states` 参数是 **delta** 还是 **累积全长**
3. 若是 **delta**：你的集成层必须以 delta 方式喂入；samd 这类"累积框架"必须维护两套状态（持久 tokens + 临时 hidden_states delta）

**反过来的等价警示**：外部 reviewer 指出 length mismatch / shape mismatch 这类 "Blocker" 时，**先确认该是不是调用约定差异**而非真 bug —— 见 [[2026-05-24-learning-external-review-blocker-triage]]。
