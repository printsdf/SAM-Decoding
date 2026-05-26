---
doc_type: decision
category: constraint
slug: phased-bench-trace-split
title: 评测脚本必须 trace OFF/ON 分两 phase 跑，不允许同次同时收加速比和 V_miss
date: 2026-05-26
status: active
area: evaluation
tags: [evaluation, vmiss, wall-time, trace, phased-benchmark, measurement-protocol]
source_feature: 2026-05-25-bench-cross-domain-speedup
---

# 评测脚本必须 trace OFF/ON 分两 phase 跑，不允许同次同时收加速比和 V_miss

## 结论

在同一个 benchmark 上同时需要 wall-time 加速比和 V_miss 时，**必须分两次 inference 跑**：

- **phase1**：不传 `--collect_diagnosis_trace`（trace OFF）→ 测 wall-time speedup_ratio
- **phase2**：显式传 `--collect_diagnosis_trace`（trace ON）→ 收 V_miss

两次 inference 使用同一份 `question.jsonl`（保证 prompt 一致），用 `_p1` / `_p2` model-id 后缀区分输出文件防止覆盖。分析时加速比取 phase1 结果，V_miss 取 phase2 结果，**不允许跨 phase 混用**。

## 背景

`--collect_diagnosis_trace` 开启后每个 decode step 多出：

1. 一次 vocab-wide `torch.argmax`（取 verifier_target token）
2. 一次 `t2d` buffer O(1) lookup（判断 token 是否在 draft 词表内）

单步开销 < 1% wall-time，但这个数字在加速比场景里不可忽略：

- sam_only 组加速比约 1.1–1.4x，~1% 增量足以令"有正收益"的结论向"几乎无收益"偏移
- 不同组 trace 状态不一致时（A 组 ON、B 组 OFF），跨组对比存在系统性方向偏差

## 考虑过的替代方案

**方案 A：trace ON 一次跑完**（被拒）
最省 GPU 时间，但加速比数据带 ~1% 上误差，且无法保证跨组比较公平。对 sam_only 等低加速比组结论影响尤其大。

**方案 B：只报 V_miss，不报加速比**（被拒）
研究目标同时需要两个维度；去掉加速比无法支撑"SAM 切换有净正收益"的结论。

## 影响与约束

- 新增 benchmark 脚本时必须遵守此分离结构；`scripts/run_bench_cross_domain_speedup.sh` 是参考实现
- phase1 / phase2 必须共用同一份 `question.jsonl`，fetch 阶段产出后两 phase 都直接读取，不重新生成
- sam_only 组（无 t2d，V_miss 无意义）只跑 phase1；`analyze_vmiss.py` 在 phase2 analyze 时复用 phase1 的 sam_only 文件，保留 N/A 行作为第三组对照
- model-id 命名约定：phase1 用 `{group}_p1`，phase2 用 `{group}_p2`

## 相关文档

- `.codestable/compound/2026-05-26-learning-phased-bench-trace-protocol.md`（learning，含"为什么不能一次跑"的详细说明）
- `.codestable/architecture/ARCHITECTURE.md` 第 5 节（已归入硬约束）
