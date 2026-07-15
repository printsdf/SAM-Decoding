# Research: Official EAGLE3 Files and Documentation

- **Query**: 寻找 README_tail_usage.md, eval_official_tail_v2.py, train_eagle3.py, llama3_eagle.py 及相关官方文件
- **Scope**: internal (codebase search)
- **Date**: 2026-06-05

## Findings

### 关键发现

用户提到的文件名在当前代码库中**并不存在**，但找到了**功能等价**或**命名相似**的实际文件。以下是映射关系：

| 用户提到的文件 | 实际存在的文件 | 说明 |
|---|---|---|
| `README_tail_usage.md` | `TAIL_README.md` | 快速参考卡片 |
| `eval_official_tail_v2.py` | `scripts/run_official_tail_v2.sh` | Shell 脚本调用 Python（见下文） |
| `train_eagle3.py` | **不存在** | 项目中无训练脚本 |
| `llama3_eagle.py` | `samd/model_patch/llama_eagle3.py` | EAGLE3 LLaMA 补丁 |

### Files Found: Tail Documentation

完整的 Tail 文档结构：

| File Path | Description |
|---|---|
| `/Users/printsdf/Research/paper_code/SAM-Decoding/TAIL_README.md` | 快速参考卡片（1页） |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/TAIL_INTEGRATION_SUMMARY.md` | 完整集成总结（5分钟） |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/TAIL_INTEGRATION_STATUS.md` | 集成状态报告 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/docs/EAGLE3_TAIL_QUICKSTART.md` | 快速入门指南（10分钟） |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/docs/EAGLE3_TAIL_USAGE.md` | 完整使用手册（30分钟） |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/docs/TAIL_FILE_STRUCTURE.md` | 文件结构说明 |

### Files Found: Official Evaluation

| File Path | Description |
|---|---|
| `/Users/printsdf/Research/paper_code/SAM-Decoding/scripts/run_official_tail_v2.sh` | 官方 tail v2 评估脚本（Shell） |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/.env.example` | 环境变量配置模板（包含 OFFICIAL_EVAL 配置） |

**重要**: `scripts/run_official_tail_v2.sh` 的第 40 行调用：
```bash
python eval_official_tail_v2.py
```

但 `eval_official_tail_v2.py` **在当前代码库中不存在**。这说明：
1. 该脚本可能在外部 EAGLE 官方仓库中
2. 或需要从其他位置复制/生成
3. Shell 脚本配置了所有必要的环境变量（行 13-39）

### Files Found: Core Implementation

| File Path | Description |
|---|---|
| `/Users/printsdf/Research/paper_code/SAM-Decoding/samd/tree_model/eagle3/tail_sidecar.py` | ⭐ Tail 核心实现（PlainTail, TuckerTail, CombinedHead） |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/samd/tree_model/eagle3/eagle3.py` | EAGLE3 集成点（line 45-52） |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/samd/model_patch/llama_eagle3.py` | LLaMA EAGLE3 补丁（用户提到的 llama3_eagle.py 的实际文件） |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/samd/samd_config.py` | 配置参数（eagle3_tail_path, eagle3_tail_type） |

### Files Found: Testing & Diagnosis Tools

| File Path | Description |
|---|---|
| `/Users/printsdf/Research/paper_code/SAM-Decoding/scripts/test_tail_simple.sh` | 简单测试脚本 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/scripts/test_tail_integration.sh` | 集成测试 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/scripts/compare_baseline_tail.sh` | 性能对比 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/scripts/demo_tail.py` | Python 演示 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/scripts/inspect_tail_checkpoint.py` | Checkpoint 检查工具 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/scripts/diagnose_tail.py` | 诊断工具 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/scripts/run_tail_eval.sh` | Tail 评估 |

### Files Found: Evaluation Scripts

所有评估脚本位于 `evaluation/` 目录：

| File Path | Description |
|---|---|
| `/Users/printsdf/Research/paper_code/SAM-Decoding/evaluation/inference_samd.py` | 主评估脚本（支持 tail） |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/evaluation/eval_llama3.py` | LLaMA3 评估 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/evaluation/eval_vicuna.py` | Vicuna 评估 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/evaluation/inference_eagle.py` | EAGLE 推理 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/evaluation/inference_eagle2.py` | EAGLE2 推理 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/evaluation/medqa_prep.py` | MedQA 数据准备 |
| `/Users/printsdf/Research/paper_code/SAM-Decoding/evaluation/medquad_prep.py` | MedQuAD 数据准备 |

### Code Pattern: tail_sidecar.py 核心结构

```python
# samd/tree_model/eagle3/tail_sidecar.py

# Line 10-17: PlainTail (低秩分解)
class PlainTail(nn.Module):
    down: nn.Linear  # hidden_size -> rank
    up: nn.Linear    # rank -> n_vmiss

# Line 20-46: TuckerTail (Tucker 分解，更节省参数)
class TuckerTail(nn.Module):
    down_c: nn.Linear     # hidden_size -> rank_c
    down_v: nn.Embedding  # n_vmiss -> rank_v
    core: nn.Parameter    # rank_c × rank_v

# Line 49-97: CombinedHead (合并 lm_head + tail)
class CombinedHead(nn.Module):
    def forward(self, x):
        logits_draft = self.lm_head_original(x)     # 32K
        logits_vmiss = self.tail_sidecar(x)         # 96K
        return torch.cat([logits_draft, logits_vmiss], dim=-1)  # 128K

# Line 152-189: attach_tail_sidecar (主附加函数)
def attach_tail_sidecar(model, tail_path, tail_type, dtype, device):
    # 1. 加载 checkpoint
    # 2. 推断类型 (plain/tucker)
    # 3. 构建 tail 模块
    # 4. 包装 lm_head -> CombinedHead
    # 5. 更新 EAGLE3 配置
```

### Code Pattern: EAGLE3 集成点

```python
# samd/tree_model/eagle3/eagle3.py:45-52
if config.eagle3_tail_path is not None:
    attach_tail_sidecar(
        self.model,
        tail_path=config.eagle3_tail_path,
        tail_type=config.eagle3_tail_type,
        dtype=dtype,
        device=device,
    )
```

### Code Pattern: 配置参数

```python
# samd/samd_config.py:40-41, 62-65
@dataclass
class SamdConfig:
    eagle3_tail_path: Optional[str] = field(default=None)
    eagle3_tail_type: Literal["auto", "plain", "tucker"] = field(default="auto")
    
    def __post_init__(self):
        if self.eagle3_tail_path is not None and self.tree_method != "eagle3":
            raise ValueError('eagle3_tail_path only supports tree_method="eagle3"')
```

### External References

从 `.env.example` 和文档推断：

- **官方 EAGLE 仓库**: 应包含 `eval_official_tail_v2.py` 和训练脚本
- **环境变量**: `OFFICIAL_EAGLE_PYTHONPATH` 用于指定官方 EAGLE 包路径
- **推荐路径**: `/path/to/EAGLE_engram:/path/to/EAGLE_engram/eagle/traineagle3`

### Related Specs

| Spec File | Description |
|---|---|
| `/Users/printsdf/Research/paper_code/SAM-Decoding/.trellis/spec/backend/eagle3-integration.md` | EAGLE3 集成规范 |

## Caveats / Not Found

### 不存在的文件

1. **`eval_official_tail_v2.py`**: 
   - Shell 脚本中被调用，但不在当前代码库
   - 可能在外部 EAGLE 官方仓库
   - 所有配置通过环境变量传递（见 `run_official_tail_v2.sh`）

2. **`train_eagle3.py` 或任何训练脚本**:
   - 项目中**完全没有训练代码**
   - 仅包含推理和评估代码
   - 训练脚本可能在：
     - 官方 EAGLE 仓库
     - 或作者的私有训练环境

3. **`README_tail_usage.md`**:
   - 不存在此精确文件名
   - 实际是 `TAIL_README.md`（功能相同）

### EAGLE 目录结构

搜索发现以下 EAGLE 相关目录：

```
evaluation/model/eagle/      # EAGLE (v1) 模型实现
samd/tree_model/eagle/       # EAGLE (v1) tree 实现
```

**但没有找到名为 `EAGLE3/` 或包含 `train` 的目录**。

### 官方脚本依赖关系

`run_official_tail_v2.sh` 配置的环境变量（行 27-39）：

```bash
OFFICIAL_EVAL_DTYPE=bfloat16
OFFICIAL_EVAL_DEVICE=cuda:0
OFFICIAL_EVAL_LIMIT_MEDQA=80
OFFICIAL_EVAL_MAX_NEW_TOKENS=256
OFFICIAL_EVAL_RUN_BASELINE=0
OFFICIAL_EVAL_RUN_MTBENCH=0
OFFICIAL_EVAL_TAIL_TYPES=auto
OFFICIAL_EVAL_OUTPUT_PATH=evaluation/data/official_tail_v2_results.json
OFFICIAL_EAGLE3_TOTAL_TOKEN=60
OFFICIAL_EAGLE3_DEPTH=7
OFFICIAL_EAGLE3_TOP_K=10
```

这些变量表明 `eval_official_tail_v2.py` 的接口设计。

## 总结

### 存在的核心文件

✅ **文档**: 6 个完整的 markdown 文档（TAIL_README, QUICKSTART, USAGE 等）  
✅ **实现**: tail_sidecar.py, eagle3.py, llama_eagle3.py  
✅ **工具**: 7 个测试/诊断脚本  
✅ **评估**: inference_samd.py 及其他评估脚本  
✅ **配置**: .env.example, samd_config.py  

### 不存在的文件

❌ **eval_official_tail_v2.py**: 被 Shell 脚本调用，但不在代码库  
❌ **train_eagle3.py**: 项目中无任何训练脚本  
❌ **README_tail_usage.md**: 应为 TAIL_README.md  

### 建议

1. 如需 `eval_official_tail_v2.py`，应从官方 EAGLE 仓库获取
2. 训练脚本不在此项目范围，需查阅官方 EAGLE3 训练代码
3. 现有文档已非常完整，可直接使用 `TAIL_README.md` 和 `EAGLE3_TAIL_QUICKSTART.md`
