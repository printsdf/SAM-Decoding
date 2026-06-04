---
doc_type: decision
category: architecture
status: active
summary: SAM + EAGLE3 融合当前默认采用 sam_sequence_graft，Stage B2 作为保守变体而非默认
tags: [sam, eagle3, tree-fusion, speculative-decoding]
---

# Decision: SAM + EAGLE3 默认融合策略采用 `sam_sequence_graft`

## 背景

本轮实验比较了几种 SAM + EAGLE3 speculative decoding 融合方式：

- pure EAGLE3
- legacy `samd_eagle3`
- Stage A `tree_fusion="sam_sequence_graft"`
- Stage B `tree_fusion="sam_tree_union_prune"`
- Stage B2 `tree_fusion="eagle_prefix_sam_expand"`

目标是在 MT-Bench 和 MedQuAD 上提高 accepted length / throughput，并降低 V_miss，同时避免 verifier 成本或候选树语义漂移导致退化。

## 决策

当前默认 / 最优 SAM + EAGLE3 融合策略采用：

```text
tree_method="eagle3"
tree_fusion="sam_sequence_graft"
```

`eagle_prefix_sam_expand` 保留为可继续探索的保守变体，但不替代 `sam_sequence_graft` 作为默认策略，除非后续重复实验或 budget sweep 显示稳定、显著收益。

## 理由

Stage A `sam_sequence_graft` 是当前最清晰的正向结果：

| Bench | Group | mean_accept | tokens/s | speedup | tree_steps | V_miss rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mt_bench | eagle3 | 5.52 | 143.4 | 3.19x | 9704 | 21.5% |
| mt_bench | samd_eagle3 | 5.60 | 148.3 | 3.30x | 8544 | 18.7% |
| mt_bench | `sam_sequence_graft` | 5.854 | 154.7 | 3.44x | 9169 | 17.1% |
| medquad | eagle3 | 4.97 | 127.5 | 2.91x | 11874 | 28.8% |
| medquad | samd_eagle3 | 4.87 | 132.7 | 3.03x | 11476 | 28.0% |
| medquad | `sam_sequence_graft` | 5.012 | 138.4 | 3.16x | 11791 | 26.9% |

Compared with legacy `samd_eagle3`, `sam_sequence_graft` improves throughput by about 4.3% on both MT-Bench and MedQuAD, while reducing V_miss.

Stage B full Dynamic-SAM tree union (`sam_tree_union_prune`, m16/k4) regressed on throughput, accepted length, and V_miss. This makes broad SAM-tree union unsuitable as the next default direction.

Stage B2 EAGLE-prefix local expansion (`eagle_prefix_sam_expand`, m4/k2) is near-neutral / slightly positive, but the improvement is too small to replace Stage A:

| Bench | Stage A mean_accept | Stage B2 mean_accept | Stage A tokens/s | Stage B2 tokens/s | Stage A V_miss | Stage B2 V_miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mt_bench | 5.854 | 5.856 | 154.7 | 154.3 | 17.1% | 17.0% |
| medquad | 5.012 | 5.014 | 138.4 | 138.1 | 26.9% | 26.9% |

## 考虑过的替代方案

### 继续使用 legacy `samd_eagle3`

未选择。它低于 `sam_sequence_graft` 的 throughput 和 V_miss 表现。

### 使用 `sam_tree_union_prune` 作为默认

未选择。首次 m16/k4 实验是负结果：MT-Bench 和 MedQuAD 的 throughput、mean_accept、V_miss 均弱于 Stage A。

### 使用 `eagle_prefix_sam_expand` 作为默认

暂不选择。它验证了保守 local expansion 不会明显退化，但相对 Stage A 的收益接近 run noise；需要重复实验或小 budget sweep 后才能提升为默认策略。

## 后续影响

- 新实验或文档中提到 SAM + EAGLE3 当前最佳默认策略时，应优先指向 `sam_sequence_graft`。
- `eagle_prefix_sam_expand` 可作为 ablation / follow-up variant，不应在没有更多证据时覆盖默认策略。
- 不应继续简单扩大 `sam_tree_union_prune` budget；若重启 broad SAM tree union，需要先重新设计 pruning objective。

## 相关文档

- `docs/experiments/results/2026-06-01-sam-eagle3-tree-fusion.md`
- `docs/experiments/results/2026-06-01-sam-eagle3-tree-union-pruning.md`
- `docs/experiments/results/2026-06-01-eagle-prefix-sam-local-expansion.md`
