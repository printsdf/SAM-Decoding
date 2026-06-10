# Phase 3.1 远端验证命令

## ✅ 本地已完成

- Phase 3.1 实现完成
- 编译检查通过
- Git commit: `09c97b5`
- 修改文件：9 个

## 📋 修改内容

### 核心修改
1. **Eagle3 返回 logprobs**:
   - `eagle3_model.py`: 收集并返回 draft_logprobs
   - `eagle3.py`: 添加 `return_logprobs=True` 参数

2. **Fusion 使用 logprobs**:
   - `naive_fusion.py`: 使用 logprobs 作为 Eagle 分数
   - `utils.py`: 传递 logprobs 到融合模块

3. **测试与文档**:
   - `test_naive_fusion.py`: 增加 logprob 对齐验证
   - `tree-fusion.md`: 文档化 logprob 契约

### 向后兼容
- Eagle3-only 使用不受影响（`return_logprobs=False` 默认）
- 其他 fusion mode 不受影响

---

## 🚀 远端执行步骤

### Step 1: 同步代码

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 拉取最新代码
git fetch origin feature/eagle3-tail-sidecar
git pull origin feature/eagle3-tail-sidecar

# 或从本地同步修改的文件
```

### Step 2: 运行测试（可选）

```bash
# 单元测试
pytest tests/test_naive_fusion.py -v

# 预期：所有测试通过
```

### Step 3: MT-Bench 完整评估

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 设置环境变量
export MODEL_PATH="/teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct"
export TREE_MODEL_PATH="/teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B"

# 运行 MT-Bench 评估（naive fusion with real logprobs）
python evaluation/inference_samd.py \
  --template llama3 \
  --model-type llama3 \
  --model-path "$MODEL_PATH" \
  --model-id mt_bench_naive_logprob \
  --bench-name mt_bench \
  --answer-file evaluation/data/mt_bench/model_answer/naive_logprob.jsonl \
  --tree_method eagle3 \
  --tree_model_path "$TREE_MODEL_PATH" \
  --fusion_mode naive \
  --fusion_max_draft_tokens 60 \
  --eagle3_total_token 60 \
  --eagle3_depth 7 \
  --eagle3_top_k 10 \
  --max_cache_len 4096 \
  2>&1 | tee logs/mt_bench_naive_logprob.log

# 查看结果
tail -100 logs/mt_bench_naive_logprob.log
```

---

## 📊 预期结果

### 性能目标

| Method | MAT | TPS | vs Eagle3 | 状态 |
|--------|-----|-----|-----------|------|
| eagle3_only | 5.52 | 143.4 | 1.000x | Baseline |
| sam_sequence_graft | 5.85 | 154.7 | 1.079x | Upper bound |
| naive (depth proxy) | < 5.0 | < 130 | < 0.91x | 失败（HumanEval 0.816x） |
| **naive (logprob)** | **> 5.3** | **> 140** | **> 0.98x** | **目标（+5-10%）** |

### 判断标准

**✅ 成功**（达到以下任一标准）:
- TPS > 140 (+2.3% vs Eagle3-only)
- MAT > 5.3 (-3.6% vs Eagle3-only)
- Eagle accept rate > 40%（vs 之前 13.7%）

**⚠️ 部分成功**:
- TPS 135-140（接近 baseline）
- Eagle accept rate 30-40%
- → 说明 logprob 有帮助，但需要 Phase 3.2（payoff 校准）

**❌ 失败**:
- TPS < 135（仍然低于 baseline）
- Eagle accept rate < 30%
- → 需要检查 logprob 提取是否正确

---

## 🔍 诊断检查

如果结果不理想，检查以下内容：

### 1. 验证 logprob 提取

在日志中查找统计输出，检查：
```
Eagle logprob range: [-15.2, -0.3]  # 应该是负数范围
Eagle avg score: -5.2               # 平均 logprob
```

如果看到奇怪的值（如全是 0 或很大的正数），说明 logprob 提取有问题。

### 2. 检查分数分布

对比 Eagle 和 SAM 的归一化分数：
```
After normalization:
  Eagle: [0.1, 0.9] distribution
  SAM:   [0.2, 0.8] distribution
```

如果分布严重不平衡，可能需要调整归一化策略。

### 3. 检查接受率

```
Acceptance rates:
  Eagle: 45.2% (目标 > 40%)
  SAM:   18.3%
```

如果 Eagle accept rate 显著提升，说明 logprob 有效。

---

## 📈 后续步骤

### 如果成功（TPS > 140）

**继续 Phase 3.2**：实现 Payoff 校准器
- 目标：统一 Eagle/SAM 分数到 P(accept)
- 预期额外提升：10-15%
- 总目标：> 150 TPS (1.046x)

### 如果部分成功（TPS 135-140）

**仍然进入 Phase 3.2**：
- Logprob 有帮助但不够
- Payoff 校准可能带来突破

### 如果失败（TPS < 135）

**调试优先**：
1. 检查 logprob 提取是否正确
2. 对比 MT-Bench 上的 Eagle3-only baseline
3. 分析 case study（哪些样本变好/变差）

---

## 🎯 成功标准总结

**Phase 3.1 成功**：
- ✅ 代码实现正确
- ✅ 编译测试通过
- ✅ 性能提升 5-10%（或至少接近 baseline）
- ✅ 为 Phase 3.2 打好基础

**Phase 3 最终目标**：
- Phase 3.1 + 3.2 + 3.3：> 157 TPS (1.095x)
- 超过 sam_sequence_graft (154.7 TPS, 1.079x)

---

## 📝 实验记录

完成评估后，请记录：
1. 完整的统计输出（Naive Fusion Analysis）
2. 性能数字（MAT, TPS, steps）
3. 与 baseline 的对比
4. 任何异常或发现

这些数据将用于：
- Phase 3.2 的设计决策
- 论文的实验结果
- 消融实验分析

---

**准备好后在远端执行，祝实验顺利！** 🚀
