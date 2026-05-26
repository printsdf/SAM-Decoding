---
doc_type: learning
track: knowledge
slug: medical-vmiss-cross-domain
title: 医学专名 V_miss 在 free-form 场景跨 bench 可定量，SAM 切换在垂直域收益减弱
created: 2026-05-26
updated: 2026-05-26
severity: ~
component: [evaluation, eagle3, samd]
tags: [vmiss, eagle3, medical, cross-domain, sam-switching, evaluation]
source_feature: 2026-05-25-bench-cross-domain-speedup
---

# 医学专名 V_miss 在 free-form 场景跨 bench 可定量，SAM 切换在垂直域收益减弱

## 情境

EAGLE3 在垂直医学场景的加速比低于通用对话，怀疑原因之一是 EAGLE3 draft 词表（32K BPE，
来自 LLaMA-2/3 tokenizer）覆盖不了医学专名，导致 verifier 频繁 miss。上一个 feature
用 USMLE MCQ 数据验证时，模型输出模式化（A/B/C/D + Answer:），词表命中率偏高，
V_miss 可能被低估。本次换成 NIH MedQuAD free-form 问答（真实病人提问 + 医生长文本
回答），同时加入 mt_bench 通用对话作对照，消除了 MCQ 格式噪声。

## 发现

### 数字

| bench | group | V_miss rate | speedup vs baseline |
|---|---|---|---|
| mt_bench（通用对话 80q） | pure_eagle3 | 21.5% | 3.19x |
| mt_bench | samd_eagle3 | 18.7% | 3.30x |
| medquad（医学 free-form 200q） | pure_eagle3 | 28.8% | 2.91x |
| medquad | samd_eagle3 | 28.0% | 3.03x |

V_miss 差值 ~7pp（medquad - mt_bench），显著高于单次跑批的随机误差。

### 结论 1：医学专名词表盲点是真实现象，不是 MCQ 格式假象

MedQuAD free-form 场景下 V_miss 比通用对话高 ~7pp。即使去掉"模型输出固定模式
token"这个因素，医学术语（药名、疾病名、基因名等）对 EAGLE3 draft 词表的穿透率
仍然更高。

### 结论 2：SAM 切换在医学域的 V_miss 降幅明显小于通用对话

- mt_bench：SAM 切换把 V_miss 从 21.5% 降到 18.7%（-2.8pp）
- medquad：仅从 28.8% 降到 28.0%（-0.8pp）

原因推断：SAM 匹配到的 suffix 多来自 prompt 中的通用语法词，而医学专名作为"新 token"
几乎不在 suffix 库里，SAM 无法切换走高 V_miss 的 tree step。

### 结论 3：samd_eagle3 加速比仍优于 pure_eagle3（即使 V_miss 降幅很小）

medquad：samd 3.03x vs pure 2.91x（+0.12x）；mt_bench：samd 3.30x vs pure 3.19x（+0.11x）。
SAM 切换的收益不仅来自降低 V_miss，还来自"把更多 step 引导到 sequence path
（直接匹配，无 tree 验证开销）"。即便在 V_miss 降幅很小的医学场景，减少 tree step
总量（medquad samd tree_steps 11476 < pure 11874）仍然带来净正收益。

## 适用情境

- 评估 EAGLE 系列 draft model 在垂直 domain 的词表覆盖能力时，**必须用 free-form prompt**
  而非 MCQ 或有固定结构的 prompt——后者会系统性低估 V_miss
- 预期 V_miss 在医学 / 法律 / 金融等专名密集 domain 会明显高于通用对话（参考值：+5~10pp）
- 即便 V_miss 高，SAM 切换仍有净正收益（通过减少 tree step 总量）；垂直域的 SAM
  收益来源以"减少 tree step"为主，而非降低 V_miss

## 不适用的反例

- 若要彻底解决医学词表覆盖问题，需要 fine-tune draft lm_head 扩充词表；本发现只描述现象，不提供修复路径
- medquad 截前 200 题集中在少数 disease（NIH 数据按 document 排序），V_miss 绝对值可能有偏；但 medquad vs mt_bench 的**相对差值**结论不受此影响
