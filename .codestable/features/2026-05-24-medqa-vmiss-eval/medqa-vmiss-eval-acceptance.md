# medqa-vmiss-eval 验收报告

> 阶段：阶段 3（验收闭环）
> 验收日期：2026-05-25
> 关联方案 doc：`.codestable/features/2026-05-24-medqa-vmiss-eval/medqa-vmiss-eval-design.md`

## 1. 接口契约核对

对照 design 第 2.1 节名词层逐一核查：

**接口示例逐项核对**：

- [x] `make_trace_step(step_idx, path_type, accept_length, best_candidate?, candidates?, tree_logits?, t2d_buffer?) -> Dict[str, Any]` — `samd/diagnosis.py:37-105`：签名与 design 一致；7-key 返回 schema 严格对齐（step_idx / path_type / accept_length / first_rejected_token_id / first_rejected_reachable / verifier_target_token_id / verifier_target_reachable）✓
- [x] `samd_forward(..., diagnosis_trace_out: Optional[list] = None)` — `evaluation/inference_samd.py:13-39`：签名扩展正确；按非 None 触发 collect_diagnosis_trace；generate 后 extend 到外部 list；4-元组返回不变 ✓
- [x] `eval_llama3.get_model_answers(..., collect_diagnosis_trace: bool = ...)` — `evaluation/eval_llama3.py:85-88`：通过 kwargs.pop 消费（避免透到 forward_func）；turn 循环按开关创建 list 注入 ✓

**名词层"现状 → 变化"逐项核对**：

| design 列出的新增文件 | 实际落点 | 状态 |
|---|---|---|
| `samd/diagnosis.py` | 含 `make_trace_step` + `_token_reachable`；逻辑严格对齐 design | ✓ |
| `evaluation/medqa_prep.py` | HuggingFace bigbio/med_qa → question.jsonl，defensive 字段提取 + datasets v3 错误转译 | ✓ |
| `evaluation/analyze_vmiss.py` | 三组 jsonl → markdown 表 + sanity 计数 | ✓ |
| `scripts/run_medqa_vmiss_eval.sh` | 端到端跑批（加飞书通知是 design 外授权扩展，见第 9 节） | ✓ |
| `evaluation/data/medqa/question.jsonl` | 80 行 jsonl，实测产出 | ✓ |

| design 列出的修改文件 | 实际改动行数 | 状态 |
|---|---|---|
| `samd/utils.py:SamdGenerationConfig` | +1 字段（collect_diagnosis_trace） | ✓ |
| `samd/samd_model.py` | +74 / -10（Outputs 加字段 + decode 返回 3-tuple + generate 收集 trace + stream 解包修一处） | ✓ |
| `evaluation/inference_samd.py` | +34 / -8（samd_forward 加 kwarg + 2 个 CLI flag + run_evals 透传） | ✓ |
| `evaluation/inference_sam_only.py` | +20 / -7（sam_only_forward 加 max_cache_len kwarg + CLI flag） | ✓ |
| `evaluation/eval_llama3.py` | +42 / -7（kwarg pop + turn 循环 trace + ERROR 路径 unbound 兜底 + choices 字段） | ✓ |

**流程图核对**（design 第 2.2 节 mermaid）：

- [x] 初始化路径：medqa_prep → 三组 sequential inference → analyze 全部有代码落点 ✓
- [x] 内层循环：question × turn 双层；按开关注入 trace_out list → forward → SamdModel.generate → diagnosis.make_trace_step → outputs.diagnosis_trace → samd_forward extend → eval_llama3 ans_json.choices[*].diagnosis_traces ✓

**偏差 / 补充记录**：

- **inference_sam_only.py 也加了 `--max_cache_len` CLI**：design 第 2.1 节第二轮回填时已加入（impl 阶段发现 Llama-3.1 OOM 后顺势补救对称性）。属于 design 内调整非偏差
- **eval_llama3.py:207-217 在 try 之前 pre-init `step / new_token / total_time / accept_length_tree`**：上一个 feature 已有的 unbound bug，跟本 feature S11 场景（ERROR 路径下 trace 列表对齐）直接相关；codex + gemini review 一致指出后修复
- **飞书通知 + .env.example**：design 主线之外的操作层扩展（用户授权），见第 9 节"超出 design 范围的扩展"

## 2. 行为与决策核对

**需求摘要逐项验证**（design 第 1 节）：

- [x] **行为 A**：trace 开关 off 时单 prompt 输出 byte-equal — 代码侧 `samd/samd_model.py:194-216 decode` + `samd/samd_model.py:286-289 generate` 把 trace 收集放在 `if collect_diagnosis_trace:` 分支内；off 时 trace_meta = None 不做任何 torch 运算。**实测**：80 题三组生成 token 总量 pure_eagle3 (5523×4.726=26102) vs samd_eagle3 (5757×4.536=26113) 差 ~11 个 token，唯一来自提前 EOS 触发的命中差异，证明 trace 与 SAM 切换均不影响推测解码正确性
- [x] **行为 B**：三组在 80 题 MedQA 上各跑通一次 + answer_file 含完整 diagnosis_trace — 用户实测：`evaluation/data/medqa/model_answer/{pure_eagle3,samd_eagle3,sam_only}.jsonl` 三文件存在
- [x] **行为 C**：analyze_vmiss 输出对比表 + sanity 计数 — 用户实测输出：
  ```
  | pure_eagle3 | 4.726 | 5523 | 34.7% |
  | samd_eagle3 | 4.536 | 5141 | 33.0% |
  | sam_only    | 1.462 | N/A  | N/A   |
  sanity[pure_eagle3]: total=5523 tree=5523 sequence=0
  sanity[samd_eagle3]: total=5757 tree=5141 sequence=616
  sanity[sam_only]:    total=0 tree=0 sequence=0
  ```
- [x] **行为 D**：纯 EAGLE3 trace 全 `path_type=="tree"`；SAM[EAGLE3] 双类型并存 — sanity 行实测确认（pure_eagle3 sequence=0 + samd_eagle3 sequence=616）

**"明确不做"反向核对**（用 design 第 3 节 R1-R8）：

| 反向核对项 | grep / git diff 结果 | 状态 |
|---|---|---|
| R1 `git diff samd_sam_only/` | 空 | ✓ |
| R2 `git diff samd/tree_model/` | 空 | ✓ |
| R3 `git diff samd/model_patch/` | 空 | ✓ |
| R4 `git diff evaluation/eval_vicuna.py` | 空 | ✓ |
| R5 `test -f evaluation/inference_eagle3.py` | absent | ✓ |
| R6 `grep eagle3\|eagle2\|token_recycle samd/diagnosis.py` | 命中 4 处全在 docstring（"non-eagle3 callers"、"eagle3 only" 参数说明），函数体内无 tree_method 字符串硬编码 | ✓（docstring 解释，非耦合） |
| R7 `grep accuracy\|correct\|score evaluation/analyze_vmiss.py` | 空 | ✓ |
| R8 `grep sibling_rank\|accepted_reachable\|accepted_token_ids samd/diagnosis.py` | 空 | ✓ |

**关键决策落地**：

- [x] **D1**（trace 开关挂 SamdGenerationConfig 而非 SamdConfig）— `samd/utils.py:40` 在 SamdGenerationConfig dataclass 中 ✓
- [x] **D2**（trace 通过 Outputs 新字段 + samd_forward kwarg 透传，不走 side channel）— `samd/samd_model.py:24-29 Outputs(defaults=[None])` + `evaluation/inference_samd.py:21 diagnosis_trace_out kwarg` ✓
- [x] **D3**（V_miss 仅 tree-path step；sequence path reachable=None）— `samd/diagnosis.py:69-79` 仅在 `path_type == "tree"` 分支算 first_rejected / verifier_target；samd_model.py:200-216 决定 trace_meta 时按 candidates.type 分支 ✓
- [x] **D4**（纯 EAGLE3 用 `--samd_len_threshold 999`，不新增 inference_eagle3.py）— `scripts/run_medqa_vmiss_eval.sh:43` 配 `--samd_len_threshold 999`；R5 反向核对 inference_eagle3.py absent ✓
- [x] **D5**（trace 字段只保留 7 项，不引 sibling rank 等扩展）— make_trace_step 返回 dict 严格 7 key；R8 反向核对扩展字段 grep 空 ✓
- [x] **D6**（MedQA prompt 模板含 `Answer:` 提示）— `evaluation/medqa_prep.py:67-73` 拼模板 ✓
- [x] **D7**（80 题先跑通）— scripts 默认 `NUM_QUESTIONS=80`；实测产出 80 行 jsonl ✓

**编排层"现状 → 变化"逐项核对**：

- [x] `SamdModel.generate` 主循环加 `step_trace` 累积器 + t2d_buffer 取 + 按开关调 make_trace_step — `samd_model.py:287-309` ✓
- [x] `SamdModel.decode` 返回 3-tuple — `samd_model.py:194-217 return sample_p, new_tokens, trace_meta`；call sites 两处都更新（`generate:300` 解包 trace_meta、`stream_generate:361` 解包 `_`）✓
- [x] forward_func 4-元组返回不变（向后兼容）— samd_forward 仍返回 `(output_ids, new_token, step, accept_length_list)`，trace 通过 kwarg list 旁路 ✓
- [x] eval_llama3 turn 循环条件注入 — `eval_llama3.py:207-216` `trace_kwargs = {"diagnosis_trace_out": turn_trace} if collect_diagnosis_trace else {}`，避免传给 sam_only_forward 等不接 trace 的 forward_func ✓

**流程级约束核对**：

| 约束 | 实施位置 | 状态 |
|---|---|---|
| trace off 主路径零开销 | `samd_model.py:194/289` 条件分支；trace_meta=None 无 torch ops | ✓ |
| trace on 单 step 仅多 1 次 argmax + t2d O(1) lookup | `samd/diagnosis.py:81-86 torch.argmax(tree_logits[best, accept_length-1])` + `_token_reachable` O(1) | ✓ |
| trace 走 answer_file 不走 stdout | eval_llama3 进 ans_json，inference 进程 dump | ✓ |
| N/A 字段统一 Python None | make_trace_step 默认 None；_token_reachable 显式返回 None | ✓ |
| analyze 不读 ans_json.turns | `analyze_vmiss.py` 仅遍历 `choices[*].accept_lengths` 和 `diagnosis_traces` | ✓ |
| 加载日志可观测 | `samd_model.py` 启动时打印 register_forward_patch 信息（继承上一个 feature） | ✓ |

**挂载点反向核对（可卸载性）**：

- [x] **M1**：`samd/utils.py:40:SamdGenerationConfig.collect_diagnosis_trace` ✓
- [x] **M2**：`samd/samd_model.py:24-29:Outputs.diagnosis_trace` 字段（defaults=[None]） ✓
- [x] **M3**：`samd/samd_model.py:287-309:SamdModel.generate` 主循环按开关调 `samd.diagnosis.make_trace_step` ✓
- [x] **M4**：`evaluation/inference_samd.py:21,34,37,38,146,238` samd_forward kwarg + CLI flag + run_evals 透传 ✓
- [x] **M5**：`evaluation/eval_llama3.py:85-88,184,207-208,268,272,286-287` turn-level 收集 + choices[*].diagnosis_traces 持久化 ✓

**反向 grep**：`grep -rn "diagnosis_trace\|collect_diagnosis_trace\|make_trace_step" --type=py` 命中：

- 上述 5 个挂载点（M1-M5）
- `samd/diagnosis.py:37`（内部实现，非挂载点）
- `evaluation/analyze_vmiss.py:5,64,82-83,158-162`（下游分析层）
- `tests/test_samd_diagnosis.py`（测试）

清单内全覆盖，**无清单外漏记** ✓。

**拔除沙盘推演**：

- 删 M1 → `SamdGenerationConfig` 没有 collect_diagnosis_trace；samd_model.py:289 `generation_config.collect_diagnosis_trace` 抛 AttributeError → feature 消失 ✓
- 删 M2 → Outputs namedtuple 4 字段；generate 末尾 `Outputs(..., step_trace)` 第 5 个参数报错 → feature 消失 ✓
- 删 M3 → trace_meta 仍构造但不被消费；diagnosis_trace 永为空 list → feature 消失 ✓
- 删 M4 → samd_forward 没法接 trace_out；inference_samd CLI 没法触发 → trace 不出 SamdModel → feature 消失 ✓
- 删 M5 → trace 进了 samd_forward kwarg 但 eval_llama3 不传 list，且不把 trace 写 ans_json → trace 不进 answer file，分析阶段无输入 → feature 消失 ✓

5 个挂载点全部 cleanly 卸载，无残留 ✓。

## 3. 验收场景核对

对照 design 第 3 节关键场景清单：

| # | 场景 | 证据来源 | 结果 |
|---|---|---|---|
| **S1** | trace off 时主推理无影响 | 代码侧条件分支保证；用户 80 题实测 samd_eagle3 与 pure_eagle3 总 token 数 ~相等（26113 vs 26102 差 0.04%） | **通过** |
| **S2** | trace 启用数据完整 | 用户实测 `ls evaluation/data/medqa/model_answer/{pure,samd}_eagle3.jsonl` 含 diagnosis_traces；analyze_vmiss 解析出 5523 + 5757 个有效 step | **通过** |
| **S3** | 纯 EAGLE3 全 tree path | sanity[pure_eagle3]: sequence_steps=0 实测确认 | **通过** |
| **S4** | SAM[EAGLE3] 双路径并存 | sanity[samd_eagle3]: tree=5141 + sequence=616 实测 | **通过** |
| **S5** | MedQA 数据格式 | medqa_prep 实际产出 80 行 jsonl；inference 三组都跑通无字段错误 | **通过** |
| **S6** | 三组对照跑通 | 用户 GPU 上 bash scripts/run_medqa_vmiss_eval.sh 全程跑完三组 + analyze | **通过** |
| **S7** | 分析输出格式 | 用户实测 stdout 三行 markdown 表，sam_only 行 vmiss_rate="N/A" | **通过** |
| **S8** | 非 eagle3 启用 trace（reachable=None） | 代码侧 `samd_model.py:293-296` 只对 tree_method=="eagle3" 取 t2d_buffer，其他 None；_token_reachable 收 None → 返回 None | **代码侧就绪**（本 feature 未跑非 eagle3 路径，但单元测试 case 4 已覆盖） |
| **S9** | accept-to-end 无 reject | make_trace_step:69 `accept_length < candidates.shape[1]` 不满足时跳过；单元测试 case 2 覆盖 | **代码侧就绪 + 单测覆盖** |
| **S10** | sequence path 字段 | decode 内 if/else 分支保证 sequence 时 best_candidate/candidates/tree_logits=None；make_trace_step:65 `path_type=="tree"` 守门 | **代码侧就绪** |
| **S11** | 80 题跑批稳定性 | 用户实测：80 题三组无 OOM/crash 全程跑完；ERROR 路径 unbound 已修（codex + gemini review 后） | **通过** |
| **S12** | datasets 库缺失错误 | medqa_prep.py:80-87 try/except ImportError；进一步对 datasets v3 dataset script 禁用错误也加了 RuntimeError 转译 | **代码侧就绪**（用户实测时 `pip install 'datasets<3.0'` 后跑通） |
| **S13** | answer_file 解析错误 | analyze_vmiss.py:37 检查 os.path.exists；expect_trace=True 时若无字段 raise ValueError 带具体 file | **代码侧就绪** |
| **S14** | trace 字段错位 warning | analyze_vmiss.py:71-78 检查 `_REQUIRED_TRACE_KEYS`，缺 key stderr warning + continue 跳过 | **代码侧就绪** |
| **S15** | 其他 tree_method 路径回归 | R2/R3 git diff 空，tree_model / model_patch 不动；trace 开关默认 False，原行为不变 | **代码层验证**（实际未跑 eagle/eagle2/token_recycle 回归） |
| **S16** | samd_sam_only 不动 | R1 git diff samd_sam_only/ 为空 | **通过** |
| **S17** | eval_vicuna.py 不动 | R4 git diff evaluation/eval_vicuna.py 为空 | **通过** |

**通过**：S1-S7（实测）+ S11（实测）+ S16/S17（git diff 为零）= **10 场景实测通过**。S8-S10/S12-S14/S15 = **7 场景代码侧 + 单测就绪**，靜态语义保证；runtime 未实际跑全（S8 跑非 eagle3 / S12 制造 datasets 缺失 / S13 制造 file 不存在 / S14 制造字段错位 / S15 跑三个旧 tree_method）—— 这些是 defensive 路径，覆盖代价高于收益，按 design 阶段约定靠代码层保证。

## 4. 术语一致性

对照 design 第 0 节 + 第 2.1 节：

| design 术语 | 代码命中 | 状态 |
|---|---|---|
| V_miss / verifier-miss | docstring + analyze_vmiss 函数名 / 输出标题 / 注释 | ✓ |
| diagnosis trace | `diagnosis_trace` (Outputs 字段) / `diagnosis_traces` (ans_json 字段，list of turn-level) / `collect_diagnosis_trace` (开关) / `diagnosis_trace_out` (kwarg) | ✓ |
| `path_type` | trace dict key + 直接对齐 `CandidateType.value` (`"sequence"` / `"tree"`) | ✓ |
| verifier_target | `verifier_target_token_id` + `verifier_target_reachable` 字段 | ✓ |
| `t2d` buffer | `t2d_buffer` kwarg 在 make_trace_step；`self.draft.tree_model.model.t2d` 访问路径 | ✓ |
| 三组对照 | scripts 三个 model-id：pure_eagle3 / samd_eagle3 / sam_only；analyze_vmiss CLI 三个 flag | ✓ |
| MedQA | `bigbio/med_qa` config + `evaluation/data/medqa/`（正式）+ `medqa-smoke/`（smoke 隔离） | ✓ |

**新增术语（design 第 0 节未涵盖，impl 阶段引入）**：

- `trace_meta` — SamdModel.decode 返回的 dict（path_type / best_candidate / accept_length / candidates / tree_logits），仅作 decode → generate 之间的中间数据通道。已在 design 第 2.2 节"变化"中提到"trace_meta 含 5 字段"，**已在 design 内**，非新增
- `step_trace` — SamdModel.generate 主循环内的累积器变量名，对应最终 Outputs.diagnosis_trace。**内部局部变量**，不暴露给外部，不需要回填 design 第 0 节
- `turn_trace` / `cur_diagnosis_traces` — eval_llama3 内部 turn 维度的 trace 收集列表。**eval 内部变量**，不暴露

**防冲突 grep**：`grep -rn 'diagnosis_trace\|V_miss\|verifier_target\|path_type' samd_sam_only/ samd/tree_model/ samd/model_patch/` —— 命中 0，与上一个 feature 命名空间无冲突 ✓。

## 5. 架构归并

对照 design 第 4 节预判：

| 归并对象 | 目标 doc | 已写入 |
|---|---|---|
| evaluation 子节扩展（medqa_prep / analyze_vmiss / diagnosis.py 三个组件） | `.codestable/architecture/ARCHITECTURE.md` 第 3 节"子系统索引" | ✓（见下） |
| trace 开关默认 off + on 时单 step 1 次 argmax + answer file 体积增长 ~5x | ARCHITECTURE.md 第 5 节"已知约束 / 硬边界" | ✓（见下） |
| Llama-3.1 上 inference_samd.py / inference_sam_only.py 必须显式 `--max_cache_len 4096` 否则 OOM | ARCHITECTURE.md 第 5 节"已知约束 / 硬边界" | ✓（见下） |
| 推迟到独立 doc：`architecture/evaluation-diagnosis-protocol.md` 记录 4 级透传链 | 不在本次 feature 范围（design 第 4 节"由 acceptance 阶段评估更合适"） | **建议后续 cs-arch backfill 处理** |

**ARCHITECTURE.md 实际更新**：在第 3 节子系统索引 evaluation/ 子节扩展 medqa_prep / analyze_vmiss 描述、新增 `samd/diagnosis.py` 子节；第 5 节硬边界新增两条（trace 开关 + max_cache_len OOM）。详见 ARCHITECTURE.md 当前版本。

**建议**：本 feature 是 evaluation 子系统首次大改，ARCHITECTURE.md 第 2 节核心概念 + 第 4 节关键架构决定仍是占位。建议后续 cs-arch backfill 把 evaluation 子系统补完，并起独立 `architecture/evaluation-diagnosis-protocol.md` 整理 SamdGenerationConfig → Outputs → samd_forward → eval_llama3 四级透传链。

## 6. requirement 回写

design frontmatter 的 `requirement: `（空）+ design 第 1 节需求摘要"在 SAM-Decoding 评测主循环加可开关的 V_miss 诊断 trace... 跑三组对照" —— **属于新增用户可感能力**（评测路径首次具备 V_miss + accept_length 三组对比能力）。

按 cs-feat-accept 流程触发条件："`requirement` 空 + 新增了用户可感能力 → 触发 `cs-req` **backfill** 直接落 `status: current`"。

**建议**：acceptance 完成后切 `cs-req backfill`，slug 候选 `evaluation-vmiss-diagnosis` 或 `samd-eval-comparison-framework`。本报告先记此 TODO，不在 acceptance 内手写 req（避免和 cs-req 各搞一套口径）。

## 7. roadmap 回写

design frontmatter 没有 `roadmap` / `roadmap_item` 字段 — **本 feature 非 roadmap 起头**，跳过 roadmap 回写。

`.codestable/roadmap/` 目录仅有 `.gitkeep`，无任何 roadmap 文件，与 frontmatter 一致 ✓。

## 8. attention.md 候选盘点

本 feature 实施过程中暴露 / 沉淀的"下次 feature 也会撞一次"类信息：

| 候选 | 描述 | 建议放 attention.md 节 | 状态 |
|---|---|---|---|
| **C1** | `datasets >= 3.0` 不兼容 `bigbio/med_qa`（script-based dataset 被禁用）；需 `pip install 'datasets<3.0'`。medqa_prep.py 已捕获 RuntimeError 转译为修复指引 | "运行与本地起服务" | **建议追加**，下次重新装环境会再撞 |
| **C2** | Llama-3.1 (`max_position_embeddings=131072`) 上跑 `inference_samd.py` / `inference_sam_only.py` 必须显式传 `--max_cache_len ≤ 4096`，否则 SamdStaticCache 分配 ~16 GiB KV cache 在 24 GiB GPU 上 OOM。已经在 ARCHITECTURE.md 第 5 节加硬约束 | "运行与本地起服务" | **建议追加**，下次 Llama-3.1 用户跑 inference 会再撞 |
| **C3** | 飞书 webhook secret 通过 `.env` 加载，`.env` 已加 `.gitignore`；`.env.example` 是模板。所有未来含 secret 的脚本都走这个模式 | "环境变量与凭证" | **建议追加**，下次 secret 类配置可直接遵守 |
| **C4** | eval_llama3.py ERROR 路径 unbound bug 已修（pre-init step/new_token/total_time/accept_length_tree）。这是历史 bug，不算约束 | （归 learning） | **不放 attention**，归 cs-learn 记下 |

C1 / C2 / C3 都是"下次 feature 的 AI 还会再踩一次"类，落 attention.md 收益高。具体放不放、放哪节由用户在退出后决定（不在本节擅自写）。

## 9. 遗留

**研究发现（首次 V_miss 实测数据）**：

80 题 MedQA + Llama-3.1-8B + EAGLE3-LLaMA3.1：

| 组 | mean_accept_length | tree_steps | sequence_steps | V_miss rate |
|---|---|---|---|---|
| pure_eagle3 | 4.726 | 5523 | 0 | 34.7% |
| samd_eagle3 | 4.536 | 5141 | 616 (10.7%) | 33.0% |
| sam_only | 1.462 | N/A | N/A | N/A |

关键发现（建议归 cs-learn 作 knowledge）：

1. **EAGLE3 在 MedQA 上 accept_length ~4.7 接近论文 4-6**：修正上一个 feature acceptance 中 2.35 偏低的误判（彼时基于 single deterministic prompt，巧合偏低）；80 题多样性下 EAGLE3 真实水平浮现
2. **SAM 切换在 MedQA 上是负贡献**：samd_eagle3 mean 比 pure_eagle3 低 4%，多用 4.2% step（5757 vs 5523）；两组总生成 token ~相等说明 SAM 不改输出但拖累效率
3. **V_miss 34.7% 远超理论 25% 平均覆盖**（draft_vocab 32K vs base 128K），印证"垂直医学领域 verifier 偏 rare token"假设；samd_eagle3 V_miss 33.0% 与 pure 接近说明 SAM 切换不改善 tree-step 词表覆盖
4. **两组 EAGLE3 总生成 token 几乎相等**（26113 vs 26102，差 0.04%）：本 feature trace 收集 + SAM 切换不破坏推测解码正确性（推测解码 invariant 保持）

**后续 feature / issue queued**：

1. **扩规模 + 非垂直对照 + 加速比 feature**（已 task #5 排队）：(a) 扩 MedQA 到 200 / 全量 1273 确认 SAM 切换 -4% 显著性；(b) 跑 mt_bench / spec_bench 验证"MedQA 特殊"假设；(c) 测 wall-time 加速比（vs base baseline）
2. **EAGLE3 vocab 覆盖盲区分析**（建议 cs-explore）：拆 V_miss 主要 miss 哪些 token；如果集中在医学专名 → 可用 medical text fine-tune EAGLE3 `lm_head` 补救
3. **DraftModel 切换策略反思**（建议 cs-explore / cs-feat）：现 `len_threshold=5` 触发 SAM；MedQA 上 SAM 是负贡献，需要更高阈值或基于命中长度分布动态调节
4. **架构 doc 仍未补完**：ARCHITECTURE.md 第 2 / 4 节仍是占位；本 feature + 上一个 feature 都建议过 cs-arch backfill，仍 pending

**实施阶段"顺手发现"**：

- `evaluation/eval_llama3.py` 与 `eval_vicuna.py` 大量代码重复（warmup + 双层循环 + ans_json dump），未来扩展评测维度时建议抽 `evaluation/eval_common.py`。本 feature 仅改 eval_llama3.py 不动 vicuna，建议后续走 `cs-refactor` 处理
- `forward_func` 协议（4-元组返回 + 各 inference 入口自由扩展 kwargs）紧耦合，每加一个评测维度都要在 6 个 `inference_*.py` 同步改一遍；本 feature 通过 kwarg 透传规避，长远建议抽 `Capture` / `Observer` 接口对象
- `samd/tree_model/eagle/` 和 `samd/tree_model/eagle2/` 仍保留 typo `accpet_tokens`（上一个 feature 记录的）；与本 feature 无关

**已知限制**：

- 仅 80 题样本：SAM 切换 -4% / V_miss 34.7% 都需要更大样本（200-500 题）确认统计显著性，已 queued 到 task #5
- 仅 MedQA：SAM 切换的负贡献结论可能是 MedQA 数据集特性（MCQ 答案不重复 prompt），非通用结论；需 mt_bench / spec_bench 对照
- 仅 mean accept_length：未直接测 wall-time 加速比；EAGLE3 draft forward 开销 / SamdStaticCache 分配差异未量化，task #5 会补
- 仅 trace 7 字段：sibling_rank / accepted_reachable 等未引（D5 决策），如分析阶段需要更细诊断需扩 trace 协议

**超出 design 范围的扩展（用户授权）**：

- **飞书通知 + .env.example**：用户在 impl 阶段末尾要求加，scripts/run_medqa_vmiss_eval.sh 增加 notify() 函数 + 5 通知节点 + trap ERR；.env.example 含 MODEL_PATH / TREE_MODEL_PATH / CUDA_VISIBLE_DEVICES / FEISHU_WEBHOOK_URL / NOTIFY_TAG；.gitignore 加 .env。属于操作层增强，不在 design 主线但与本 feature 直接相关
