# EAGLE3 Tail 文件结构

## 目录树

```
SAM-Decoding/
│
├── 文档
│   └── .trellis/tasks/06-06-dual-draft-fusion/docs/reference/
│       ├── EAGLE3_TAIL_QUICKSTART.md       # 快速入门指南
│       ├── EAGLE3_TAIL_USAGE.md            # 完整使用手册
│       ├── TAIL_FILE_STRUCTURE.md          # 本文件
│       └── eagle3-usage.md                 # EAGLE3 开发者指南
│
├── 🔧 核心实现（已存在）
│   └── samd/
│       ├── samd_config.py                  # 配置：eagle3_tail_path/type + EAGLE3 tree 参数
│       ├── tree_model/
│       │   └── eagle3/
│       │       ├── tail_sidecar.py         # ⭐ Tail 核心实现
│       │       │   ├── PlainTail           # 低秩分解
│       │       │   ├── TuckerTail          # Tucker 分解
│       │       │   ├── CombinedHead        # 合并输出
│       │       │   └── attach_tail_sidecar # 附加函数
│       │       ├── eagle3.py               # 集成点（line 45-52）
│       │       ├── eagle3_model.py         # EAGLE3 模型
│       │       └── eagle3_config.py        # 配置类
│       └── model_patch/
│           └── llama_eagle3.py             # LLaMA EAGLE3 补丁
│
├── 🎮 CLI & 测试（已存在）
│   ├── evaluation/
│   │   └── inference_samd.py               # 评估脚本（支持 tail）
│   ├── samd/inference/
│   │   └── cli.py                          # CLI 工具（支持 tail）
│   └── tests/
│       └── test_samd.py                    # 单元测试（支持 tail）
│
├── 🛠️ 脚本工具
│   └── scripts/
│       ├── 已存在：
│       │   ├── test_samd_eagle3.sh         # EAGLE3 测试脚本
│       │   ├── inspect_tail_checkpoint.py  # Checkpoint 检查工具
│       │   └── diagnose_tail.py            # 诊断工具
│       └── 新增：
│           ├── test_tail_simple.sh         # 简单测试脚本
│           ├── compare_baseline_tail.sh    # 性能对比脚本
│           └── demo_tail.py                # Python 演示脚本
│
└── ⚙️ 配置
    ├── .env.example                        # 环境变量模板
    │   ├── EAGLE3_TAIL_PATH               # Tail checkpoint 路径
    │   └── EAGLE3_TAIL_TYPE               # Tail 类型（auto/plain/tucker）
    └── README.md                           # 主文档（已更新）
```

## 核心文件说明

### 🔧 核心实现

#### `samd/tree_model/eagle3/tail_sidecar.py` ⭐
**最重要的文件**，包含所有 tail 实现：

```python
PlainTail           # Line 10-17:  低秩分解 hidden → rank → V_miss
TuckerTail          # Line 20-46:  Tucker 分解（更节省参数）
CombinedHead        # Line 49-97:  合并 lm_head + tail → full vocab
attach_tail_sidecar # Line 152-189: 主附加函数
```

**关键流程：**
1. 加载 checkpoint
2. 推断类型（plain/tucker）
3. 构建 tail 模块
4. 包装 lm_head → CombinedHead
5. 更新 EAGLE3 配置

#### `samd/tree_model/eagle3/eagle3.py:45-52`
**集成点**，在 EAGLE3 初始化时调用 tail：

```python
if config.eagle3_tail_path is not None:
    attach_tail_sidecar(
        self.model,
        tail_path=config.eagle3_tail_path,
        tail_type=config.eagle3_tail_type,
        dtype=dtype,
        device=device,
    )
```

#### `samd/samd_config.py`
**配置参数**：

```python
eagle3_tail_path: Optional[str] = field(default=None)
eagle3_tail_type: Literal["auto", "plain", "tucker"] = field(default="auto")
eagle3_total_token: int = field(default=60)
eagle3_depth: int = field(default=7)
eagle3_top_k: int = field(default=10)

# 验证逻辑
if self.eagle3_tail_path is not None and self.tree_method != "eagle3":
    raise ValueError('eagle3_tail_path only supports tree_method="eagle3"')
```

### 🎮 CLI & 测试

所有现有的 CLI 和测试脚本都已支持 tail：

| 文件 | Tail 支持位置 |
|------|--------------|
| `evaluation/inference_samd.py` | Tail + EAGLE3 tree 参数 |
| `samd/inference/cli.py` | Tail + EAGLE3 tree 参数 |
| `tests/test_samd.py` | Tail + EAGLE3 tree 参数 |
| `scripts/test_samd_eagle3.sh` | Tail + EAGLE3 tree 参数 |

### 🛠️ 工具脚本

#### 已存在的工具

**`scripts/inspect_tail_checkpoint.py`**
- 检查 checkpoint 结构
- 验证维度匹配
- 显示训练元数据

**`scripts/diagnose_tail.py`**
- 诊断性能问题
- 对比预期结果
- 给出调试建议

#### 新增工具

**`scripts/test_tail_simple.sh`**
- 一键测试脚本
- 自动检查 checkpoint
- 环境变量驱动

**`scripts/compare_baseline_tail.sh`**
- Baseline vs Tail 对比
- 自动计算提升百分比
- 生成详细报告

**`scripts/demo_tail.py`**
- 完整 Python 示例
- 端到端演示
- 路径自动验证

### 📖 文档结构

```
文档层次（从简到详）：
1. TAIL_README.md               # 快速参考（1页）
2. TAIL_INTEGRATION_SUMMARY.md  # 集成总结（5分钟）
3. EAGLE3_TAIL_QUICKSTART.md    # 快速入门（10分钟）
4. EAGLE3_TAIL_USAGE.md         # 完整文档（30分钟）
5. TAIL_INTEGRATION_STATUS.md   # 状态报告（详细）
```

## 使用路径

### 路径 1：最快上手
```
.env.example → .env → scripts/test_tail_simple.sh
```

### 路径 2：理解原理
```
TAIL_README.md → EAGLE3_TAIL_QUICKSTART.md → tail_sidecar.py
```

### 路径 3：深度使用
```
EAGLE3_TAIL_USAGE.md → demo_tail.py → 自定义代码
```

### 路径 4：问题排查
```
diagnose_tail.py → compare_baseline_tail.sh → EAGLE3_TAIL_USAGE.md
```

## 代码依赖关系

```
用户代码
    ↓
SamdConfig (配置 tail 参数)
    ↓
Eagle3.__init__()
    ↓
attach_tail_sidecar() ← 核心函数
    ↓
├─ _infer_tail_type()      # 检测类型
├─ _build_plain_tail()     # 构建 PlainTail
├─ _build_tucker_tail()    # 构建 TuckerTail
└─ CombinedHead()          # 包装 lm_head
    ↓
Eagle3Model.lm_head = CombinedHead
    ↓
推理时自动使用 full vocab
```

## 关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| PlainTail 实现 | `tail_sidecar.py` | 10-17 |
| TuckerTail 实现 | `tail_sidecar.py` | 20-46 |
| CombinedHead | `tail_sidecar.py` | 49-97 |
| attach 函数 | `tail_sidecar.py` | 152-189 |
| EAGLE3 集成点 | `eagle3.py` | 45-52 |
| 配置参数 | `samd_config.py` | 40-41 |
| CLI 参数 | `cli.py` | 284-285 |
| 评估脚本参数 | `inference_samd.py` | 163-164 |

## 快速检索

### 我想...

**...了解如何使用**
→ `EAGLE3_TAIL_QUICKSTART.md`

**...理解实现原理**
→ `EAGLE3_TAIL_USAGE.md` + `samd/tree_model/eagle3/tail_sidecar.py`

**...测试 tail 功能**
→ `scripts/test_tail_simple.sh` 或 `scripts/demo_tail.py`

**...对比性能**
→ `scripts/compare_baseline_tail.sh`

**...检查 checkpoint**
→ `scripts/inspect_tail_checkpoint.py`

**...解决问题**
→ `scripts/diagnose_tail.py` + `EAGLE3_TAIL_USAGE.md`（故障排查章节）

**...修改代码**
→ `samd/tree_model/eagle3/tail_sidecar.py`（核心逻辑）

**...添加新 tail 类型**
→ 修改 `tail_sidecar.py`，参考 PlainTail/TuckerTail 实现

## 总结

- ✅ **已存在** 8 个核心文件（无需修改）
- 🆕 **新增** 7 个文档和工具文件
- 📝 **更新** 2 个文件（README.md, .env.example）
- 🎯 **使用** 只需配置 2 个参数

所有文件都已就绪，可以直接使用！
