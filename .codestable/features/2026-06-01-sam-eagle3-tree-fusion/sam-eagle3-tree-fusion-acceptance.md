---
doc_type: feature-acceptance
feature: 2026-06-01-sam-eagle3-tree-fusion
design: sam-eagle3-tree-fusion-design.md
checklist: sam-eagle3-tree-fusion-checklist.yaml
requirement: sam-eagle3-tree-fusion
status: accepted
accepted_at: 2026-06-01
tags: [sam, eagle3, tree-fusion, speculative-decoding]
---

# sam-eagle3-tree-fusion acceptance

## 1. 验收结论

Stage A 的代码形态和架构落点已经对齐 approved design：主路径新增 `tree_fusion="sam_sequence_graft"`，在 `tree_method="eagle3"` 时先保留完整 EAGLE3 tree，再把命中阈值的 raw SAM sequence 作为 shared-prefix branch graft 到同一棵验证树中，最终仍以 `CandidateType.tree` 交给现有 verifier 消费。

本次验收是**源码级通过 + ablation 通过 + greedy equivalence 由 owner 豁免**：目标服务器已完成 MT-Bench / MedQuAD 上的 fusion p1（trace OFF）测速与 p2（trace ON）诊断。fusion 在两个数据集上相对 legacy SAMD+EAGLE3 都提高 tokens/s 与 mean accepted tokens，并降低 V_miss。由于本 feature 未改 target verifier / posterior 接受规则，owner 确认不把单独 greedy equivalence 实跑作为合并阻塞项；该豁免不等同于独立 correctness proof。

## 2. 已完成范围

- `samd/samd_config.py`
  - 新增 `SamdConfig.tree_fusion`，默认 `"none"`。
  - 校验 `"sam_sequence_graft"` 只支持 `tree_method="eagle3"`，其他 tree method 明确报错。
- `samd/tree_model/fusion.py`
  - 新增内部 `TreeSpec(tokens, parents)` 表示。
  - 支持从 EAGLE3 tree buffers 反推父子关系。
  - 支持导出 `tree_attn_mask / tree_position_ids / tree_retrieve_indices`。
  - 支持 `graft_sequence()`，按同父同 token 复用 shared prefix，只追加缺失 suffix。
- `samd/sam/dyn_sam.py`、`samd/sam/static_sam.py`
  - 新增 `gen_draft_raw()`，用于返回不含 legacy padding `0` 的 SAM sequence。
  - 原有 `gen_draft()` padding 行为不变。
- `samd/draft.py`
  - 在 `DraftModel.lookup()` 接入 fusion branch。
  - fusion 模式下总是先调用 `tree_model.gen_draft(start_token)`，保持 EAGLE3 增量状态机被正常推进。
  - SAM match 不足时返回纯 EAGLE3 tree；SAM match 达阈值时返回 fused tree。
- `samd/inference/cli.py`、`evaluation/inference_samd.py`
  - 新增 `--tree_fusion` 参数并透传到 `SamdConfig`，支持三组 ablation 使用同一入口对比。

## 3. Checklist 状态

已标记 `passed` 的源码级 / 结构级检查：

- `SamdConfig.tree_fusion` 默认值和配置合法性。
- 非 EAGLE3 开启 `sam_sequence_graft` 明确报错。
- fusion 位于 `DraftModel` / tree model 输出组合层，不侵入 `Eagle3Model.topK_genrate()`。
- fused candidate 仍返回 `CandidateType.tree`，接口形态可被现有 `SamdModel.decode()` / `eval_posterior()` 消费。
- Stage A 保留完整 EAGLE3 tree，只做 shared-prefix graft，不做 score pruning。
- raw SAM sequence 不接入 legacy padding token。
- 默认 `tree_fusion="none"` 仍走 legacy 二选一路径。
- SAM match 不足时返回纯 EAGLE3 tree。
- SAM match 达阈值时 fused tree 包含 EAGLE3 tree 与 SAM branch 真实 token。
- shared prefix 场景复用已有节点，不新增同父同 token 重复节点。
- 未新增 `tree_method="eagle3_fusion"`。
- 未把 `samd_sam_only` 的 `cnt_endpos / states_topk_next` multi-branch SAM tree 接入主 `samd`。
- 未根据 EAGLE3 score 或 SAM score 删除 EAGLE3 节点。

已补齐并标记 `passed` 的运行级检查：

1. 最小 ablation 三组实跑，并收集 decode steps / accepted length per step / total decode tokens / wall-time / speedup ratio。

由 owner 确认豁免的运行级检查：

1. 开启 fusion 后 greedy 输出仍与 base model greedy 一致。

## 4. 验证证据

### 4.1 静态 / 轻量验证

- 语法检查目标文件：
  - `samd/samd_config.py`
  - `samd/tree_model/fusion.py`
  - `samd/draft.py`
  - `samd/sam/dyn_sam.py`
  - `samd/sam/static_sam.py`
  - `samd/inference/cli.py`
  - `evaluation/inference_samd.py`
- YAML 检查目标文件：
  - `.codestable/features/2026-06-01-sam-eagle3-tree-fusion/sam-eagle3-tree-fusion-checklist.yaml`
- 范围守护检查：
  - `eagle3_fusion` 未出现在 Python 代码中。
  - `cnt_endpos|states_topk_next` 未出现在主 `samd/` 路径中。
  - `Eagle3Model.topK_genrate()` 未接入 SAM / fusion 逻辑。

### 4.2 源码核对

- `SamdConfig.__post_init__()` 在加载 tree config 前先校验 `tree_fusion`，能阻止错误组合静默进入运行阶段。
- `DraftModel.lookup()` 在 fusion branch 中先生成 EAGLE3 draft tree，再根据 SAM match 决定是否 graft，因此不会绕开 EAGLE3 的 pending hidden states / stable KV 推进约束。
- `TreeSpec.from_eagle3_buffers()` 只从 EAGLE3 buffers 恢复结构；`graft_sequence()` 基于 `parents[index] == parent and tokens[index] == token` 做 prefix 复用；`to_buffers()` 再导出标准 tree buffers。
- `DynSAM.gen_draft_raw()` 与 `StaticSAM.gen_draft_raw()` 都只切片真实输入 token，不调用 legacy padding 分支。

### 4.3 运行验证

服务器完成了 MT-Bench / MedQuAD 上的 pure EAGLE3、legacy SAMD+EAGLE3、fusion 三组对比。测速遵守既有约束：p1 trace OFF 用于 wall-time / speedup；p2 trace ON 用于 tree step count / V_miss，不混用 p2 wall-time 做速度结论。

| Bench | Group | mean_accept | tokens/s | speedup | tree_steps | V_miss rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| mt_bench | eagle3 | 5.52 | 143.4 | 3.19x | 9704 | 21.5% |
| mt_bench | samd_eagle3 | 5.60 | 148.3 | 3.30x | 8544 | 18.7% |
| mt_bench | fusion | 5.854 | 154.7 | 3.44x | 9169 | 17.1% |
| medquad | eagle3 | 4.97 | 127.5 | 2.91x | 11874 | 28.8% |
| medquad | samd_eagle3 | 4.87 | 132.7 | 3.03x | 11476 | 28.0% |
| medquad | fusion | 5.012 | 138.4 | 3.16x | 11791 | 26.9% |

Fusion 相对 legacy SAMD+EAGLE3：

- MT-Bench：mean_accept +4.5%，tokens/s +4.3%，speedup +0.14x，V_miss -1.6 pp。
- MedQuAD：mean_accept +2.9%，tokens/s +4.3%，speedup +0.13x，V_miss -1.1 pp。

详细结果记录见 `docs/experiments/results/2026-06-01-sam-eagle3-tree-fusion.md`。

## 5. Greedy equivalence 豁免说明

未单独跑 base greedy vs fusion greedy 的逐题输出对比。原因：本 feature 没有修改 `eval_posterior()`、target model forward 的接受规则或 greedy sampling 规则，只改变 draft candidate 的组织方式；owner 确认该项不作为本阶段 acceptance 阻塞项。

保留的工程风险是：candidate tree buffer / cache selection 若有 bug，仍可能间接影响输出。因此后续若要写成强 correctness claim 或论文复现实验，建议再取小 subset 补一轮输出对比。

## 6. 文档回写

- 已回写 `.codestable/architecture/ARCHITECTURE.md`：记录 `samd/tree_model/fusion.py`、`tree_fusion` 与 `tree_method` 分层、以及 DraftModel 层的 SAM + EAGLE3 candidate 编排。
- 已 backfill `.codestable/requirements/sam-eagle3-tree-fusion.md`：记录“同一步同时验证 SAM 检索候选和 EAGLE3 draft tree”的当前能力。
- 已更新 `.codestable/requirements/VISION.md`：把 `sam-eagle3-tree-fusion` 加入 current 能力列表。

## 7. 后续建议

- 如果要写论文级 correctness 声明，补一轮小 subset greedy equivalence；否则当前 Stage A 可以按 accepted 进入后续实验。
- 如果继续扩展，再为 Stage B 单独开 design：SAM multi-branch tree union + pruning。Stage B 不应直接在本 feature 上顺手扩展。
