---
doc_type: requirement
slug: evaluation-vmiss-diagnosis
pitch: 在自己数据集上对比推测解码策略的真实加速度，并暴露 draft 模型的词表盲点
status: current
last_reviewed: 2026-05-26
implemented_by:
  - 2026-05-24-medqa-vmiss-eval
  - 2026-05-25-bench-cross-domain-speedup
tags: [evaluation, vmiss, accept-length, benchmark, draft-model]
---

# 在自己数据集上对比推测解码策略的真实加速度

## 用户故事

- 作为推测解码研究员，我想知道 SAM[EAGLE3] 在 MedQA、法律、金融这种垂直数据集上比纯 EAGLE3 快多少，而不是只看论文在通用 benchmark 上 claim 的加速比。
- 作为想用 EAGLE3 加速医学问答的工程师，我想知道 verifier 在我的 domain 上期望的 token 多少落在 EAGLE3 draft 词表外（即 V_miss），决定要不要 fine-tune draft lm_head。
- 作为论文实验作者,我想一条命令把三组对照（纯 EAGLE3 / SAM-only / SAM[EAGLE3]）跑完 + 自动出对比表，而不是手动拼 6 个 inference 入口再 grep 算统计。
- 作为评测开发者，我希望诊断 trace 默认 off 不影响生产推理性能，只在研究 / benchmark 时按需 on。

## 为什么需要

推测解码论文经常 claim 5-7x 加速，但通用 benchmark（mt_bench / spec_bench）的结论很难直接套到垂直领域。研究员想知道自己数据集（医学 / 法律 / 金融问答）的真实加速比，得自己拼三组对照脚本、自己写统计代码。而且看到"加速比不达论文 claim"时也不知道原因 —— 可能是 draft 模型词表覆盖不了垂直领域的术语（verifier 想要的下一个 token 根本不在 draft vocab 内），可能是 SAM 切换在 prompt 后缀短的场景没发力，但没有分维度的诊断数据就只能瞎猜。

## 怎么解决

把"verifier 拒收 draft 时它想要哪个 token、那个 token 在不在 draft 词表内"这件事做成 inference 主循环的可开关旁路 trace。默认关闭，对主推理路径零影响。开启时每个解码步骤记一条 trace，最后随 answer file 一起落盘。配套提供 MedQA 数据准备脚本 + 三组对照一键跑批入口 + 自动分析脚本：跑完一条命令，看到三组的平均 accept_length 对比 + tree-step 上的 V_miss 比例 + SAM 切换占比。

## 边界

- 只看加速度和词表覆盖维度，不评估答题正确率 —— "模型答对没"是另一回事。
- 默认 trace 开关 off，不动生产推理性能；只在研究 / benchmark 时 on。
- 只有 EAGLE3 路径能算 V_miss（依赖它的 draft 词表映射机制）；其他 tree_method（EAGLE / EAGLE2 / Token Recycle）的 trace 字段会留空。
- 不提供"看到 V_miss 高之后怎么补救"的能力 —— fine-tune draft model 词表、扩词表都是另外的事。
- 只支持 Llama 系列 base model + 单 GPU + batch_size=1（继承 samd 现有限制）。

## 变更日志

### 2026-05-26（feature `2026-05-25-bench-cross-domain-speedup`）

- 扩展到 2 个 bench（NIH MedQuAD 200 题 free-form 医学问答 + mt_bench 80 题通用对话）
- 加 wall-time 加速比维度：4 组对照（新增 transformers baseline）× 2 bench；phase1 trace OFF 保证速度数据纯净
- 首次跨 bench 定量验证 V_miss 差异：medquad 28.8% vs mt_bench 21.5%（~7pp），确认医学专名词表盲点不是 MCQ 输出模式导致的假象
- 加速比结论：samd_eagle3 在 mt_bench 3.30x / medquad 3.03x，均优于 pure_eagle3（3.19x / 2.91x），SAM 切换在两个 domain 均有净正收益

### 2026-05-25（feature `2026-05-24-medqa-vmiss-eval`）

- 初始实现：diagnosis trace 基础设施（7-key schema）+ MedQA MCQ 80 题 + 三组对照一键跑批 + analyze_vmiss.py
