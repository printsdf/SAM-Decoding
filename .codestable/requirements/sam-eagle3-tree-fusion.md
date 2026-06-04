---
doc_type: requirement
slug: sam-eagle3-tree-fusion
pitch: 同时验证 SAM 检索候选和 EAGLE3 draft tree，让推测解码更容易吃到长接受段
status: current
last_reviewed: 2026-06-01
implemented_by:
  - 2026-06-01-sam-eagle3-tree-fusion
tags: [sam, eagle3, tree-fusion, speculative-decoding, accept-length]
---

# 同时验证 SAM 和 EAGLE3 候选

## 用户故事

- 作为做 SAM-Decoding 组合实验的研究员，我希望 SAM 的检索候选和 EAGLE3 的 draft tree 能在同一步里一起验证，而不是每次只能二选一。
- 作为追求推理加速的人，我希望在 SAM 命中长匹配时保留 EAGLE3 的候选覆盖，同时追加 SAM 的长分支，增加每步接受更多 token 的机会。
- 作为做 ablation 的实验作者，我希望能通过一个开关比较 EAGLE3、原本的 SAMD+EAGLE3 二选一、以及 tree fusion 三组，而不是维护多套推理入口。

## 为什么需要

原来的 SAMD+EAGLE3 路径在每个 decode step 里只选一个 draft 来源：SAM match 足够长就走 SAM sequence，否则走 EAGLE3 tree。这个策略简单，但会错过两类候选的互补性：SAM 可能给出长的上下文复用分支，EAGLE3 则提供模型预测的多分支覆盖。研究者想验证二者叠加是否能提高 accepted tokens / step，需要一个不改 verifier 语义、可和旧路径并排比较的融合能力。

## 怎么解决

系统增加一个候选树组合开关。默认行为不变；开启 SAM sequence graft 后，EAGLE3 仍先生成完整 tree，SAM 命中阈值时把不含 padding 的 SAM sequence 作为额外分支 graft 到同一棵验证树中。最后仍交给原有 verifier 一次 forward 验证，因此输出语义不变，只改变候选组织方式。

## 边界

- 只支持 EAGLE3 路径；其他 draft tree model 不会静默启用这项能力。
- 第一版只 graft 一条 SAM sequence，不做完整的 SAM multi-branch tree fusion。
- 不根据 SAM 或 EAGLE3 的分数剪掉 EAGLE3 节点；这项能力优先验证候选互补性。
- 不改变最终 greedy decoding 语义；target model 的接受规则仍是唯一判定标准。
- 不解决 Static SAM 数据来源问题；离线语料和 artifact 构建属于独立能力。

## 变更日志

- 2026-06-01：backfill 当前能力，记录 `tree_fusion="sam_sequence_graft"` 已接入 EAGLE3 路径。
