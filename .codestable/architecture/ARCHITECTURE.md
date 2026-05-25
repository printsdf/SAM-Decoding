# SAM-Decoding 架构总入口

> 状态：骨架（待补全 — 见第 2/4 节占位）
> 创建日期：2026-05-23
> 最近更新：2026-05-25（feature `2026-05-24-medqa-vmiss-eval` 归并）

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
  - **`samd/diagnosis.py`** — per-step diagnosis trace 工具（2026-05-24 引入）。`make_trace_step` 构造 7-key trace dict（含 `path_type` / `accept_length` / `first_rejected_*` / `verifier_target_*`），`_token_reachable` 用 EAGLE3 `t2d` buffer 判 draft 词表覆盖。函数纯净，对 tree_method 不耦合（通过 `t2d_buffer` 参数解耦，非 eagle3 callers 收 None 退化）。`SamdGenerationConfig.collect_diagnosis_trace=True` 时由 `SamdModel.generate` 每 decode step 调用一次
- **`samd_sam_only/`** — 不带 draft model 的 SAM-only 优化版本，不消费 `tree_method`
- **`evaluation/`** — Spec-Bench / MT-Bench / MedQA 风格 benchmark 入口：
  - `inference_samd.py` — samd 主路径 benchmark 入口，原生支持 `--tree_method ∈ {token_recycle, eagle, eagle2, eagle3}` + `--collect_diagnosis_trace` + `--max_cache_len`
  - `inference_sam_only.py` — SAM-only 路径 benchmark 入口，支持 `--max_cache_len`
  - `inference_{baseline,eagle,eagle2,pld,token_recycle}.py` — 各路径独立入口
  - `eval_llama3.py` / `eval_vicuna.py` — `forward_func` 协议主循环（question × turn 双层），按 `collect_diagnosis_trace` kwarg 在 ans_json `choices[*].diagnosis_traces` 写 per-turn trace
  - `medqa_prep.py` — HuggingFace `bigbio/med_qa` → Spec-Bench 格式 `question.jsonl`（2026-05-24 引入，需 `datasets<3.0`）
  - `analyze_vmiss.py` — 三组 answer file → V_miss / accept_length 对比 markdown 表（2026-05-24 引入）
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
- **`SamdGenerationConfig.collect_diagnosis_trace` 默认 False**（2026-05-24 引入）。off 时主推理路径零开销（trace_meta=None，无 torch ops）；on 时单 decode step 多 1 次 vocab-wide `torch.argmax`（取 verifier_target）+ 1 次 `t2d` O(1) lookup（< 1% wall-time 增量）；answer file 体积增长 ~5x（per-step trace dict 7 字段持久化）
- **Llama-3.1 / 长 context base model 跑 `evaluation/inference_samd.py` 或 `inference_sam_only.py` 时必须显式传 `--max_cache_len ≤ 4096`**（2026-05-24 引入约束）。默认值是 `model.lm.config.max_position_embeddings`（Llama-3.1 = 131072），`SamdStaticCache.__init__` 会 alloc `[B × kv_heads × max_cache_len × head_dim × 2(K+V) × dtype_bytes × num_layers]` = ~16 GiB KV cache，与模型自身 16 GiB + EAGLE3 draft 1 GiB 一起远超 24 GiB GPU。`tests/test_samd.py` 走 CLI `--max_cache_len` 默认 2048 不受影响；只有 inference 入口需要补

