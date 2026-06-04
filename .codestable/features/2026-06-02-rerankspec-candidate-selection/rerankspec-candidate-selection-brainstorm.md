---
doc_type: feature-brainstorm
feature: 2026-06-02-rerankspec-candidate-selection
status: archived
summary: RerankSpec candidate selection idea 已迁出本项目，后续应在更适合的多候选 speculative decoding 项目中验证
tags: [rerankspec, speculative-decoding, sam, candidate-selection, oracle-gap]
---

# RerankSpec Candidate Selection Brainstorm

## 归档说明

- 归档日期：2026-06-02
- 归档状态：迁出本项目；本仓库不继续承载 RerankSpec oracle-gap pilot 代码。
- 原因：本项目的标准 Dynamic SAM 是 single-path proposer，而真正需要验证的是 multi-candidate proposer 的 candidate selection gap；后续应在更适合的项目中用 SAM-only Static SAM tree / Suffix-style / n-gram / REST-style proposer 做系统实验。
- 迁出时保留的结论：不要把早期 Dynamic-SAM-derived pilot 的 0 oracle gap 当作反证；那组结果只说明当时 collector framing 错了。
- 下一步：在新项目中从“candidate selection bottleneck”出发，先做 top-M oracle-gap analysis，再决定是否实现 acceptance-aware reranker。

> Stage 0 | 2026-06-02 | 下一步：oracle-gap pilot / design

## 想做什么、为什么

当前想法不是简单给 SAM / Suffix 候选加一个 LightGBM 或 MLP reranker，而是先把论文问题定义为：model-free speculative decoding 的隐藏瓶颈可能不只是“找不到候选”，而是“top-M 候选里已有更好 continuation，但 heuristic selector 选错了”。

如果能系统量化这个 candidate selection gap，并证明一个低开销、lossless-verifier-compatible 的 acceptance-aware selector 能吃掉一部分 gap，这个方向才有机会从工程 trick 上升为 ACL/EMNLP 级别的系统性发现。

## 考虑过的方向

### 方向 A：简单 learned reranker

- 描述 / 价值 / 代价：SAM 或 Suffix top-M 候选后接 LightGBM / MLP，直接预测哪个候选更好；实现快，但容易被 reviewer 认为只是小 trick 或普通 acceptance predictor。
- 结论：否决作为论文主线；可以作为后续方法实例之一。

### 方向 B：candidate selection oracle-gap study

- 描述 / 价值 / 代价：先离线枚举 top-M retrieved candidates，对每条候选用 target model teacher-forced verifier 计算 accepted length，比较 baseline selection 和 oracle-best。若 gap 明显，就能证明“选候选”本身是独立瓶颈。
- 结论：选定为第一阶段；这是 Go/No-Go gate。

### 方向 C：plug-in acceptance-aware reranking framework

- 描述 / 价值 / 代价：把 reranker 写成 proposer-agnostic 框架，特征包含 match length、frequency、source distance、recent acceptance、draft length / utility cost 等；可接 SAM、Suffix、n-gram、REST-style retrieval。实验更重，但创新性更稳。
- 结论：作为第二阶段候选；只有 oracle gap 足够大才推进。

## 已敲定的设计点

- 已确认：论文 framing 围绕 “candidate selection bottleneck”，不要围绕 “add a reranker”。
- 已确认：第一阶段只做 offline oracle-gap pilot，不先训练 reranker。
- 已确认：pilot 的核心指标是 baseline accepted length、oracle-best accepted length among top-M、oracle gap、positive-gap step rate、candidate count。
- 已确认：最终输出分布必须由 target model lossless verification 保证；selector 只改变 proposal choice。
- 倾向：标题使用 “Beyond Longest Match: Acceptance-Aware Candidate Selection for Model-Free Speculative Decoding”。
- 待验证：SAM-only 的 Static SAM tree / leaf paths 是否真的包含明显优于默认 tree path selection 的候选；标准 Dynamic SAM 只有 single-path continuation，不作为 top-M 候选选择主问题。
- 待验证：top-M enumeration 和后续 reranker overhead 是否小到能转化为 end-to-end tokens/sec gain。

## 选定方向与遗留问题

选定方向是先写一个最小 oracle-gap proof-of-concept：在 greedy decoding 过程中用已有 evaluation `question.jsonl` 和已有 Static SAM artifact 走 SAM-only tree proposer，枚举 tree leaf candidates，然后对每个候选用 target model teacher forcing 计算 path-specific greedy accepted length，输出 per-step jsonl 和 summary。Dynamic SAM 只作为 single-path sanity check，不用于证明 top-M selection gap。

明显不做：本阶段不训练 LightGBM / MLP，不接真实 serving，不改 `SamdModel.decode` 主路径，也不 claim speedup。只有当 oracle gap 足够大时，下一阶段才设计 acceptance-aware reranker 和 end-to-end speed evaluation。

遗留问题：

1. oracle-best 比 baseline 是否有足够差距（建议 Go gate：mean oracle gap 或 oracle-best 提升至少约 20%）。
2. gap 是否只在 Dynamic SAM 上出现，还是也能推广到 Static SAM / Suffix-style / n-gram proposer。
3. candidate source frequency、match length、source distance、recent acceptance 等轻量特征能否解释 oracle winner。
