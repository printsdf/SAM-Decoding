---
doc_type: decision
category: constraint
date: 2026-05-24
slug: transformers-version-pin
status: active
area: samd
tags: [transformers, dependency, version-pin, monkey-patch]
---

## 背景

samd 通过 monkey patch 改写 `transformers` 库的 `LlamaModel.forward` / `LlamaForCausalLM.forward` / `LlamaModel._update_causal_mask` 实现 tree mask 注入和 hidden state 收集。`samd/model_patch/llama.py` 和 `samd/model_patch/llama_eagle3.py:llama_model_forward_eagle3` 都基于 **transformers 4.46.3 的 LlamaModel.forward 拷贝**改造（加 EAGLE3 收集逻辑）。

`transformers` 在 4.50+ 之后对 `LlamaModel.forward` 做了大量重构：
- `StaticCache` 从 `modeling_llama` 移到 `cache_utils`（import 路径变）
- `_update_causal_mask` 替换为 `create_causal_mask` 函数
- 引入 `position_embeddings` 由 LlamaModel 自己创建并传给 decoder_layer
- LlamaModel.forward 的参数列表变化

## 决定

samd 项目锁定 **`transformers == 4.46.x`**（README 已注明）。`requirements.txt` 应该 pin 该版本。

## 理由

- monkey patch 内容跟具体 transformers 版本强耦合 —— 跨 minor 版本（4.46 → 4.50+）的 API 变更会让 patch 失效或行为错乱
- 拷贝过来的 LlamaModel.forward 模板是 4.46.3 的；如果环境装的是 4.50+，patched LlamaModel.forward 期待的 `_update_causal_mask` 方法在 base LlamaModel 上根本不存在（被 `create_causal_mask` 取代），patch 执行会 silent 调到错误的 mask 路径或 AttributeError
- 不只是 EAGLE3 一处受影响 —— eagle2 path 的 `samd/model_patch/llama.py` 早就基于 4.46.x 写，跨版本一样断

## 考虑过的替代方案

- **追新 transformers，按需更新 patch**：被拒。研究项目优先复现性，每次 transformers 升级都要重新审查 patch 内 ~80 行拷贝代码的成本太高
- **改用 forward hook 取代 monkey patch**：被拒。一致性、生命周期复杂度问题，[[2026-05-24-learning-external-review-blocker-triage]] 提到的 D2 拒绝过

## 后果

- 用户安装环境必须严格按 README 走 `pip install transformers==4.46.1`（4.46.3 也兼容）
- 如果项目要支持新 transformers，本约束需要新发 decision（supersede）+ 同步重写 `samd/model_patch/llama.py` 和 `samd/model_patch/llama_eagle3.py`
- attention.md 已记录此约束（"运行与本地起服务" 节），下次任何 samd 流程 AI 启动时都会读到

## 相关文档

- README.md（实验环境节）
- attention.md（运行节）
- 代码：`samd/model_patch/llama.py`、`samd/model_patch/llama_eagle3.py`
