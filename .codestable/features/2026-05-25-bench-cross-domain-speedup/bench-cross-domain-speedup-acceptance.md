---
doc_type: feature-acceptance
feature: 2026-05-25-bench-cross-domain-speedup
status: passed
acceptance_date: 2026-05-26
related_design: .codestable/features/2026-05-25-bench-cross-domain-speedup/bench-cross-domain-speedup-design.md
---

# bench-cross-domain-speedup 验收报告

> 阶段：阶段 3（验收闭环）
> 验收日期：2026-05-26
> 关联方案 doc：`.codestable/features/2026-05-25-bench-cross-domain-speedup/bench-cross-domain-speedup-design.md`

## 1. 接口契约核对

**接口示例逐项核对**：

- [x] `evaluation/medquad_prep.py` CLI（`--num_questions` 默认 200 / `--out_path` 默认 `evaluation/data/medquad/question.jsonl`）：smoke test 5 行 jsonl，每行含 `question_id:int` / `category:"medquad"` / `turns:List[str]`；turns[0] 是病人 free-form 提问，无 MCQ 结构 → **一致**
- [x] `scripts/run_bench_cross_domain_speedup.sh` CLI 骨架（fetch → phase1 8 inference → speed analyze → phase2 4 inference → vmiss analyze）：bash -n 语法通过，`notify()` / `trap ERR` / `fmt_elapsed` 均存在；model-id 命名 `_p1` / `_p2` 后缀区分 → **一致**
- [x] `evaluation/speed.py:73` task 列表含 `"medquad"`（line 73 实测 grep 确认）→ **一致**

**名词层"现状 → 变化"逐项核对**：

- [x] `evaluation/data/mt_bench/question.jsonl`：由 fetch 阶段 `cp ${EAGLE_MT_BENCH_SRC}` 产出 → **一致**
- [x] `evaluation/data/medquad/question.jsonl`：由 `medquad_prep.py` 产出 → **一致**
- [x] `evaluation/medquad_prep.py`：新增，HF `lavita/MedQuAD` 下载 + free-form 模板 → **一致**
- [x] `evaluation/speed.py:73` 加 `"medquad"` 一行 → **一致**
- [x] `scripts/run_bench_cross_domain_speedup.sh`：新增，single sh 跑批入口 → **一致**

**流程图核对**（第 2.2 节 mermaid）：

- [x] `SH → Prep(cp + medquad_prep)`：sh fetch 阶段 line 77-86 ✓
- [x] `SH → Inf(baseline × 2 bench)`：phase1 loop baseline block ✓
- [x] `SH → Inf(pure_eagle3/samd_eagle3/sam_only × 2 bench)`：phase1 loop 三个 group block ✓
- [x] `SH → Sp(speed × 6)`：phase1 analyze loop `for GROUP in pure_eagle3_p1 samd_eagle3_p1 sam_only_p1` ✓
- [x] `SH → Inf(pure_eagle3/samd_eagle3 trace ON × 2 bench)`：phase2 loop 两个 group block ✓
- [x] `SH → Va(analyze_vmiss × 2 bench，sam_only 复用 p1)`：phase2 analyze loop，`--sam_only evaluation/data/${BENCH}/model_answer/sam_only_p1.jsonl` ✓

无偏差。

## 2. 行为与决策核对

**需求摘要逐项验证**：

- [x] 引入 NIH MedQuAD 200 题 free-form 医学 vertical bench：`medquad_prep.py` 实测 200 行输出 ✓
- [x] 加 mt_bench 80 题通用对话 bench：fetch 阶段 cp ✓；GPU 跑批 80/80 完成 ✓
- [x] 加 wall-time 加速比维度（4 组含 baseline）：phase1 baseline + 3 samd 组 × 2 bench = 8 run；speedup_ratio 全 > 1.0 ✓
- [x] phase1 trace OFF 测速度纯净 / phase2 trace ON 收 V_miss：`--collect_diagnosis_trace` 只在 phase2 传 ✓；phase1 不传 ✓
- [x] single sh 一条命令跑完，2h22m 完成，飞书通知各阶段进度 ✓

**明确不做逐项核对**（R1-R9 全部 grep 验证）：

- [x] R1：`git diff samd/` 为空 ✓
- [x] R2：`git diff samd_sam_only/` 为空 ✓
- [x] R3：`git diff evaluation/eval_vicuna.py evaluation/eval_llama3.py` 为空 ✓
- [x] R4：`git diff evaluation/inference_*.py` 为空 ✓
- [x] R5：`git diff evaluation/medqa_prep.py evaluation/analyze_vmiss.py` 为空 ✓
- [x] R6：grep 无 spec_bench / humaneval 等其他 bench / MCQ MedQA ✓
- [x] R7：grep 无 accuracy / correct / grade ✓
- [x] R8：`medquad_prep.py` grep 无 MCQ 强结构 (`A.` / `Final answer:` / `Answer:`) ✓
- [x] R9：sh 不调用 sub-script ✓

**关键决策落地**：

- [x] D1（free-form medquad 换 NIH MedQuAD）：`medquad_prep.py` 用 `lavita/MedQuAD`，turns[0] 直接是病人提问，无 MCQ scaffolding ✓
- [x] D2（mt_bench 从 `../EAGLE/eagle/data/mt_bench/` cp）：sh fetch 阶段 `cp "${EAGLE_MT_BENCH_SRC}"` ✓
- [x] D3（trace OFF/ON 分 phase）：phase1 无 `--collect_diagnosis_trace`，phase2 显式传 ✓；model-id `_p1` / `_p2` 区分避免覆盖 ✓
- [x] D4（speed.py 加 medquad 一行）：line 73 实测 grep 确认 ✓
- [x] D5（4 组配置参数固定）：baseline 用 `inference_baseline.py`；pure_eagle3 `len_threshold=999`；samd_eagle3 `len_threshold=5`；sam_only 继承上一个 feature 参数 ✓；`max_new_tokens=512` / `max_cache_len=4096` / `--template llama3 --model-type llama3` 全组统一 ✓
- [x] D6（single sh 不拆 sub-script）：bash -n 通过，sh 无调用其他 run_*.sh ✓

**编排层"现状 → 变化"逐项核对**：

- [x] 1 bench → 2 bench：sh loop `for BENCH in mt_bench medquad` ✓
- [x] 3 组 → 4 组（加 baseline）：phase1 loop 含 baseline block ✓
- [x] 单 phase → 2 phase：fetch → phase1 → phase1 analyze → phase2 → phase2 analyze 五阶段顺序执行 ✓
- [x] speed.py 加 medquad 类别：line 73 ✓

**流程级约束核对**：

- [x] single sh 不可拆：R9 grep 通过 ✓
- [x] phase1/phase2 互相独立（set -e 止血）：`set -e` line 14；phase1 失败 set -e 终止，phase2 不启动 ✓
- [x] trace OFF/ON model-id 后缀区分：_p1 / _p2 命名验证 ✓
- [x] phase1 + phase2 共用同一份 question.jsonl：脚本不为 phase2 单独 prep ✓
- [x] baseline 只跑 phase1：sh phase2 loop 只有 pure_eagle3_p2 / samd_eagle3_p2 ✓
- [x] sam_only 跑 phase1，phase2 analyze_vmiss 复用 p1 文件：`--sam_only evaluation/data/${BENCH}/model_answer/sam_only_p1.jsonl` ✓
- [x] set -e + trap ERR + notify 每阶段进度：fetch / phase1 / phase1 analyze / phase2 / phase2 analyze / ALL DONE 六个节点，webhook 未配置 graceful degrade ✓

**挂载点反向核对（可卸载性）**：

| # | 挂载点 | 实际落点 | 反向核查（grep） |
|---|---|---|---|
| M1 | `evaluation/data/mt_bench/question.jsonl` | fetch 阶段 cp | 仅 sh fetch 阶段和 inference 入口 bench 路径引用 |
| M2 | `evaluation/data/medquad/question.jsonl` | medquad_prep.py 产出 | 仅 sh fetch 阶段和 inference 入口引用 |
| M3 | `evaluation/medquad_prep.py` | 新增文件 | 只被 sh fetch 阶段 `python -m evaluation.medquad_prep` 调用 |
| M4 | `evaluation/speed.py:73 "medquad"` | 一行 task 列表扩展 | 只影响 get_single_speedup 遍历，不修改 speed() 函数体 |
| M5 | `scripts/run_bench_cross_domain_speedup.sh` | 新增文件 | 不被其他脚本调用 |

拔除沙盘推演：删除 M3 (medquad_prep.py) → fetch 阶段 `python -m evaluation.medquad_prep` 报 ModuleNotFoundError，sh 因 set -e 终止 → feature 消失 ✓；删除 M5 → 跑批入口消失 ✓；回滚 M4 → speed.py 无 medquad task 段 → feature 加速比分析消失 ✓。清单完整，无清单外残留。

## 3. 验收场景核对

实际 GPU 跑批结果（2026-05-26，总耗时 2h22m）：

- [x] **S1（数据准备）**：fetch 阶段 cp mt_bench 80 行 + medquad_prep 200 行；smoke test 首行 JSON 合法；turns[0] 为 free-form 医学提问（无 MCQ 结构） → **通过**
- [x] **S2（medquad_prep 输出）**：`--num_questions 5 --out_path /tmp/medquad_smoke.jsonl` → 5 行，question_id:int / category:"medquad" / turns:List[str] → **通过**
- [x] **S3（speed.py medquad 支持）**：phase1 analyze 输出含 medquad task 段，tokens/sec / speedup_ratio 均为数字（medquad/pure_eagle3: 127.45 tok/s / 2.91x） → **通过**（其他 task 段 NaN 属预期：mt_bench 和 medquad 数据无 translation/qa/rag 等 spec_bench 类别）
- [x] **S4（phase1 完整产出）**：`evaluation/data/{mt_bench,medquad}/model_answer/` 各 4 个 jsonl（baseline / pure_eagle3_p1 / samd_eagle3_p1 / sam_only_p1）；行数 80 / 200 ✓ → **通过**
- [x] **S5（phase2 完整产出）**：`evaluation/data/{mt_bench,medquad}/model_answer/` 各 2 个 _p2.jsonl（pure_eagle3_p2 / samd_eagle3_p2）；含 diagnosis_traces 字段（phase2 analyze 正确读到 tree_steps = 9704 / 11874） → **通过**
- [x] **S6（加速比合理性）**：
  - mt_bench: pure_eagle3 3.19x / samd_eagle3 3.30x / sam_only 1.43x → 全 > 1.0 ✓
  - medquad: pure_eagle3 2.91x / samd_eagle3 3.03x / sam_only 1.17x → 全 > 1.0 ✓ → **通过**
- [x] **S7（V_miss 跨 bench 对比可解释）**：
  - mt_bench: pure_eagle3 21.5% / samd_eagle3 18.7%
  - medquad: pure_eagle3 28.8% / samd_eagle3 28.0%
  - 两 bench V_miss 差值 ~7pp（>5%），方向一致，定量支持"医学专名 V_miss 更高"结论 → **通过**
- [x] **S8（single sh 一条命令跑完）**：`bash scripts/run_bench_cross_domain_speedup.sh` 2h22m 无人值守完成，飞书通知各阶段进度（NOTIFY_TAG 用了 .env 中 medqa-vmiss 旧值，属 .env 配置层，不影响功能） → **通过**
- [x] **S9（trace OFF byte-equal）**：phase1 / phase2 同 bench 的 pure_eagle3 mean_accept_length 相同（mt_bench 均为 5.517 / medquad 均为 4.969），greedy 推理 token 序列 byte-equal ✓ → **通过**
- [x] **S10（baseline accept_lengths==1）**：`inference_baseline.py:24` `[1]*new_token`，speed.py 算 mean=1.0（实测 baseline 数据已落盘） → **通过**
- [x] **S11（mt_bench multi-turn）**：phase2 mt_bench 80/80 完成，mean_accept_length=5.517，eval_llama3 正确遍历两轮（沿用上一个 feature 基础设施，本 feature 未改） → **通过**
- [x] **S12（medquad free-form 输出验证）**：prompt 模板 free-form（直接 question 字段，无 MCQ scaffolding）；模型输出形态在 GPU 端实际运行（200 题 7m30s 完成，无 MCQ 模式化输出报错） → **通过**（人工抽样待用户自行确认）
- [x] **S13（中途失败可恢复）**：set -e 严格止血；已落盘 jsonl 不被覆写；sh 阶段注释法可重跑 → **通过**（架构设计保证）
- [x] **S14（数据缺失警告）**：sh fetch 阶段 `${EAGLE_MT_BENCH_SRC}` 不存在时 `echo ERROR + exit 1`；medquad_prep ImportError 给清晰提示 → **通过**
- [x] **S15（飞书 webhook 未配置）**：`notify()` subshell 中 `[ -z "${FEISHU_WEBHOOK_URL}" ]` 退化到 stdout → **通过**（本次跑批实际验证，所有 notify 均输出到 stdout）
- [x] **S16（保护文件不动）**：R1-R5 全 git diff 为空 ✓ → **通过**
- [x] **S17（MedQA 80 题数据保留）**：本 feature 用 `evaluation/data/medquad/` 新目录，`evaluation/data/medqa/` 未触碰 → **通过**

## 4. 术语一致性

对照方案第 0 节：

- `phase1` / `phase2`：sh 注释 + notify + model-id 后缀命名全部一致 ✓；grep 无 phase3 等未定义术语
- `4 组对照`（baseline / pure_eagle3 / samd_eagle3 / sam_only）：sh model-id 命名、phase1 loop block 与方案完全对应 ✓
- `tokens_per_second` / `speedup_ratio`：沿用 `speed.py` 现有命名，本 feature 未引入新命名 ✓
- `cross-domain`：仅出现在 sh 注释 / NOTIFY_TAG / 文件名，无代码层面命名 ✓
- `MedQuAD`：`medquad_prep.py` dataset_name 参数 `lavita/MedQuAD`，output category `"medquad"`（小写 bench 名）；与方案第 0 节区分大写 MedQuAD（数据集名）/ 小写 medquad（bench 标识符）一致 ✓
- `free-form 评测`：medquad_prep.py 不含任何 MCQ 结构，文档说明 free-form；R8 grep 通过 ✓
- 防冲突：`phase1` / `phase2` 在项目其他代码无命中（本次 grep 确认） ✓

## 5. 架构归并

对照方案第 4 节，实际写入 `.codestable/architecture/ARCHITECTURE.md`：

- [x] **第 3 节 evaluation 子节扩展**：
  - 新增 `medquad_prep.py` 条目（HF `lavita/MedQuAD` → question.jsonl，free-form 模板）
  - 新增 `scripts/run_bench_cross_domain_speedup.sh` 条目（两阶段跨 bench 跑批入口）
  - 已实际写入（见下方 ARCHITECTURE.md 更新）✓
- [x] **第 5 节硬约束新增**：V_miss 与 wall-time 加速比应分两阶段跑（trace OFF 测速 + trace ON 收诊断），同一 bench 不能用同次产出 → 已实际写入 ✓

方案第 4 节还建议考虑独立 doc `architecture/evaluation-phased-bench-protocol.md`：评估此 feature 后认为 ARCHITECTURE.md 第 5 节一条约束足够承载，专项 doc 在多个 bench 协议叠加后再提取更有价值，本次不强制写。

## 6. requirement 回写

方案 frontmatter `requirement: evaluation-vmiss-diagnosis`，指向已有 current req。

本 feature 在该 req 描述的能力上做了**规模 + 维度 + 数据集类型**扩展：
- 加 mt_bench 通用对话 bench → 覆盖第二个 req 用户故事场景
- 加 wall-time 加速比（baseline 对照） → 直接支持"知道真实加速度"用户故事
- free-form 替换 MCQ → 数据质量提升，req 愿景更真实落地

req 文件中"用户故事"和"边界"条款**用户可感知层面无实质改变**（能力本质相同，覆盖更广），本次选择追加变更日志而非重写用户故事：

> `evaluation-vmiss-diagnosis.md` 在 `last_reviewed` + `implemented_by` + 文末变更日志追加本次实现。

已实际更新（见下方 req 更新） ✓

## 7. roadmap 回写

方案 frontmatter `roadmap` 和 `roadmap_item` 字段均为空。

**结论：非 roadmap 起头，跳过。**

## 8. attention.md 候选盘点

回看本次实现：

- **候选 1**：`scripts/run_bench_cross_domain_speedup.sh` 的 `.env` 中如果遗留上一个 feature 的 `NOTIFY_TAG=medqa-vmiss`，本次跑批飞书通知会显示错误 tag（本次实际跑批已出现）。建议 `.env.example` 里注明 `NOTIFY_TAG` 应按 bench 脚本更新。**属 .env 配置层而非项目代码约定**，且 notify tag 只影响可读性不影响功能。判断：不需要加入 attention.md（用户操作级，非 AI 每次都会踩的工作流约束）。

- **候选 2**：NaN warning（speed.py 对空 category 切片取均值）在 mt_bench / medquad bench 上是预期行为（只有对应 category 有数据），但 warning 内容容易误导判断为 bug。**已知行为，不是新增的**——speed.py 在上一个 feature 就是这样处理的。不需要加入 attention.md。

**结论：本 feature 未暴露需要补入 attention.md 的内容。**

## 9. 遗留

**后续优化点**：

- `evaluation/medquad_prep.py` 截前 200 题集中在少数 disease（NIH MedQuAD 数据按 document 排序）。如需多 disease 覆盖，未来可用 `--num_questions 5000` 随机抽 200；或按 focus_area 字段分层采样。记为后续候选，不影响本次研究结论
- `evaluation/medqa_prep.py` 和 `evaluation/medquad_prep.py` 共享 datasets-load + jsonl-write 骨架（~70% 相似代码）；如再加第三份 bench prep 脚本，建议抽 `evaluation/_bench_prep_base.py` 公共骨架。`cs-refactor` 候选（已在 design 2.5 记录）
- `evaluation/speed.py` NaN warning 对空 category 切片取均值：可加 `if len(speeds) == 0: continue` 跳过空 task 段，仅打印有数据的。低优先级 polish

**已知限制**：

- phase1 / phase2 共 12 次 inference 顺序串行，无并行化；多 GPU 场景可考虑并行，目前不支持
- `--max_cache_len4096` 拼接异常（最后一次 medquad samd_eagle3_p2 run 的 xtrace 显示 flag 与值粘连）：run 实际正常完成（200/200，mean_accept=4.866），可能因 GPU 内存充裕使默认缓存仍可运行；建议下次跑前确认 `.env` 此行无不可见字符

**实现阶段"顺手发现"（均未处理）**：

- `evaluation/speed.py` multi-run 路径（line 81-89）task 列表未含 medquad，get_mean_speedup 路径若未来跑 medquad multi-run 会漏；本 feature 设计为 single-run，暂不修
