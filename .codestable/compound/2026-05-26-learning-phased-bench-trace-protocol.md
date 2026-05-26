---
doc_type: learning
track: knowledge
slug: phased-bench-trace-protocol
title: V_miss 与加速比评测必须 trace OFF/ON 分阶段跑，同一 bench 不能混用同次产出
created: 2026-05-26
updated: 2026-05-26
severity: ~
component: [evaluation]
tags: [evaluation, vmiss, wall-time, trace, phased-benchmark, measurement]
source_feature: 2026-05-25-bench-cross-domain-speedup
---

# V_miss 与加速比评测必须 trace OFF/ON 分阶段跑，同一 bench 不能混用同次产出

## 情境

同时需要两个维度的评测数据：(1) wall-time 加速比（speedup_ratio），(2) V_miss rate。
最简单的做法是 trace ON 跑一次，既收 wall_time 也收 diagnosis_traces。

## 为什么不能这样做

`--collect_diagnosis_trace` 开启时，每个 decode step 多出：

- 1 次 vocab-wide `torch.argmax`（取 verifier_target token）
- 1 次 `t2d` buffer O(1) lookup（判断词表覆盖）

实测 < 1% wall-time 增量，听起来可忽略。但加速比的分子分母都很敏感：

- sam_only 加速比 ~1.1–1.4x，~1% 增量足以把方向性结论从"有收益"翻到"几乎没收益"
- 若某组 trace ON、某组 trace OFF，对比时存在系统性偏差，无法做跨组比较

## 正确做法

```
phase1：trace OFF（不传 --collect_diagnosis_trace）→ 测 wall-time 加速比
phase2：trace ON（显式传 --collect_diagnosis_trace）  → 收 V_miss
```

model-id 用 `_p1` / `_p2` 后缀区分，防止两阶段输出文件互相覆盖。
phase2 分析时，sam_only（不产生 trace）复用 phase1 文件作为第三组，保留 N/A 行。

工程落点：`scripts/run_bench_cross_domain_speedup.sh` 的 phase1/phase2 分离结构
是这条协议的参考实现。

## 不适用的反例

- 只关心 V_miss、不关心加速比的场景可以一次 trace ON 跑完
- 加速比差值远大于 1%（如 pure_eagle3 3x+ 这类大幅提升）时，trace overhead 可忽略；
  但保持分阶段跑是好习惯，未来数字对比更干净
