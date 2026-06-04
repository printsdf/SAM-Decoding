---
doc_type: requirement
slug: static-sam-offline-corpus
pitch: 用可信的垂直领域数据复现 Static SAM 数据库
status: draft
last_reviewed: 2026-05-29
implemented_by: []
tags: [static-sam, corpus, medical, finance, reproducibility]
---

# 用可信的垂直领域数据复现 Static SAM 数据库

## 用户故事

- 作为一个做 SAM-Decoding 垂直领域实验的人，我希望 Static SAM 来自医疗、金融这类非评测数据，而不是把 benchmark 数据混进加速实验里。
- 作为一个要复现实验的人，我希望看到每个 Static SAM artifact 用了哪些数据、哪个模型生成回复、哪个 tokenizer 构建，而不是只能拿到一个看不出来源的 `.pkl`。
- 作为一个以后还会加新领域数据的人，我希望先把数据整理成统一格式，再按不同模型构建 artifact，而不是每次复制一套临时脚本。

## 为什么需要

Static SAM 的效果很依赖离线文本库。如果数据来源说不清，或者和评测数据混在一起，实验结果就很难解释。垂直领域实验尤其需要把“数据从哪来、是否评测集、怎么变成 SAM”讲清楚，否则加速收益可能被质疑为数据泄漏或不可复现。

## 怎么解决

先把医疗、金融等 instruction/QA 数据整理成统一的离线语料库，记录来源、领域、许可和评测排除理由；再按需要生成模型风格回复，并用指定 tokenizer 构建 Static SAM artifact。语料库本身保持模型无关，artifact 才绑定具体模型。

## 边界

- 不负责自动收集所有公开医疗、金融数据；第一优先级是让用户本地数据可导入、可校验、可复现。
- 不把常见 benchmark/eval 数据集放进 Static SAM，即使它们有 train split。
- 不评估医学或金融问答准确率；这项能力只保证 Static SAM 数据构建过程可信、可复现。
- 不改变 SAM-Decoding 的推理算法；它只提供更清楚的数据和 artifact 构建入口。
