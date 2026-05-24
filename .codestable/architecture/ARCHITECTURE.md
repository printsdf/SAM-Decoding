# SAM-Decoding 架构总入口

> 状态：骨架（待补全 — 见第 2/4 节占位）
> 创建日期：2026-05-23
> 最近更新：2026-05-24（feature `2026-05-23-eagle3-integration` 归并）

## 1. 项目简介

基于 Suffix Automaton 的推测解码方案，通过对 prompt / 文本库做最长后缀匹配生成 draft，并可与 EAGLE / EAGLE2 / EAGLE3 / Token Recycle 等 draft-model-based 方法组合，按匹配长度自动切换 draft 来源。

## 2. 核心概念 / 术语表

（待 `cs-arch backfill` 补全）

## 3. 子系统 / 模块索引

- **`samd/`** — SAM-Decoding 主实现，含 SAM 切换 + 可插拔 tree_model
  - **`samd/tree_model/`** — Draft 模型集成层，按 `samd_config.tree_method` 路由到具体实现：
    - `token_recycle/` — Token Recycle 算法
    - `eagle/` — EAGLE-1 集成
    - `eagle2/` — EAGLE-2 集成
    - `eagle3/` — EAGLE-3 集成（2026-05-23 引入）。Draft 模型用 `fc(3H→H)` + 单层 `LlamaDecoderLayeremb` + 独立 `lm_head(H→draft_vocab_size)` + d2t/t2d 词表映射。需要 base model 输出 low/mid/high 三层 hidden state 拼接（3H）作为 forward 输入；与 EAGLE-2/EAGLE-1 路径并列独立切换。详见 `features/2026-05-23-eagle3-integration/eagle3-integration-design.md`
  - **`samd/model_patch/`** — base model 的 monkey patch 集合。`llama.py` 提供通用 patch（tree_mask 注入 + 返回 `last_hidden_states` 的 ModelOutput），`llama_eagle3.py` 提供 EAGLE3 专属 patch（patched `LlamaModel.forward` 在 idx ∈ {2, N//2, N-3} 三层收集 hidden states 并 concat 成 3H）。`SamdModel.register_forward_patch` 按 `tree_method` 选择对应 patch_dict
- **`samd_sam_only/`** — 不带 draft model 的 SAM-only 优化版本，不消费 `tree_method`
- **`evaluation/`** — Spec-Bench / MT-Bench 风格 benchmark 入口，`evaluation/inference_samd.py` 是 samd 主路径的 benchmark 入口，已原生支持 `--tree_method ∈ {token_recycle, eagle, eagle2, eagle3}`
- **`tools/`** — 离线工具，主要用于构建 Static SAM 数据（`prepare_prompts.py` / `gen_response.py` / `gen_sam_alpaca.py` 三步流水线）

## 4. 关键架构决定

（待 `cs-arch backfill` 补全）

## 5. 已知约束 / 硬边界

- **`tree_method="eagle3"` 时 base model 必须装上 `eagle3_patch_dict` / `eagle3_attn_patch_dict`**（由 `SamdModel.register_forward_patch` 自动按 tree_method 路由），否则 `LlamaModel.forward` 不会收集 low/mid/high 三层 hidden state，下游 Eagle3 draft 拿到的是 H 维而非 3H 维，`fc(3H→H)` 投影 shape mismatch 会立即报错；如绕过该校验则 draft 推理 silent 退化
- **EAGLE3 官方权重不含 `embed_tokens.weight`**（draft 共享 base 的 embedding）。`Eagle3.__init__` 末尾从 `lm.model.embed_tokens.weight` 复制补齐；若 base 和 draft 的 `vocab_size` 不一致，复制跳过并打印警告（draft 会以零张量推理，输出无意义）
- **EAGLE3 hidden state 抽取硬编码三层 index `{2, N//2, N-3}`**（与 `../EAGLE/eagle/model/modeling_llama_kv.py:1138` 训练侧一致）。base model 层数 N ≥ 6 才安全（否则 index 重合）
- **transformers 版本必须固定到 4.46.x**（README 注明）。`samd/model_patch/llama.py` 和 `samd/model_patch/llama_eagle3.py` 都基于 transformers 4.46.3 的 `LlamaModel.forward` 拷贝构造，4.50+ 把 `StaticCache` 移出 `modeling_llama` 且改造了 forward API
- **`SamdModel` 仅支持 `batch_size=1`**（`samd_model.py:240` 显式 assert）
- 不支持非 Llama backbone（Qwen2/Qwen3/Mixtral 等）—— 现有 patch 只覆盖 `LlamaForCausalLM` / `LlamaModel`

