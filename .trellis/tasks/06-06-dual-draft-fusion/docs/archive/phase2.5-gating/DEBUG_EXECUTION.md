# Debug 版本远端执行清单

## ✅ 本地已完成

在 `samd/utils.py` 添加了 3 行 debug 日志：
- Line ~120: 打印每次门控的输入（token, match quality, threshold）
- Line ~123: 打印 SKIP 决策
- Line ~163: 打印 FUSE 决策

## 🚀 远端执行步骤

### Step 1: 同步修改后的代码

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 备份当前版本
cp samd/utils.py samd/utils.py.before_debug

# 从本地复制修改后的 samd/utils.py
# 或用 git pull（如果已推送）
```

### Step 2: 快速测试（3 样本）

```bash
python evaluation/inference_samd.py \
  --model_type llama \
  --model_path /teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct \
  --model_id llama31-8b-instruct \
  --tree_model_path /teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B \
  --bench_name humaneval \
  --question_begin 0 \
  --question_end 3 \
  --num_choices 1 \
  --max_new_tokens 512 \
  --dtype bfloat16 \
  --fusion_mode naive \
  --answer_file ./humaneval_debug_3.jsonl \
  2>&1 | tee debug_3.log

# 立即查看门控决策
echo "=== Gate Decisions ==="
grep "\[GATE\]" debug_3.log | head -30

# 统计
echo ""
echo "=== Summary ==="
echo "Total gate checks: $(grep -c '\[GATE\] token=' debug_3.log)"
echo "SKIP decisions: $(grep -c '\[GATE\] -> SKIP' debug_3.log)"
echo "FUSE decisions: $(grep -c '\[GATE\] -> FUSE' debug_3.log)"
```

### Step 3: 分析结果

**预期格式**:
```
[GATE] token=123, dyn=2, static=1, best=2, threshold=5
[GATE] -> SKIP SAM (best=2 < 5)
[GATE] token=456, dyn=8, static=3, best=8, threshold=5
[GATE] -> FUSE SAM (best=8 >= 5)
...
```

**判断标准**:

| SKIP 比例 | 结论 | 下一步 |
|-----------|------|--------|
| > 70% | 质量门控正常，但其他问题 | 检查 Eagle-only 分支或统计 |
| 30-70% | 门控有效但不够强 | 提高 threshold 或实施方案 B |
| < 30% | 动态 SAM 质量普遍好 | 直接实施方案 B（Eagle 优先级） |

### Step 4: 根据结果调整

#### 情况 A: SKIP > 70%

**说明**: 质量门控有效，但 Eagle accept rate 仍低

**可能原因**:
1. Eagle-only fallback 分支实现有问题
2. 统计计算错误
3. SKIP 时的 Eagle tree 本身质量不高

**诊断**: 对比 SKIP 样本和 FUSE 样本的 Eagle accept rate

#### 情况 B: SKIP 30-70%

**说明**: 门控部分有效，需要更激进的策略

**方案 1**: 提高 threshold
```bash
# 测试 threshold=10
python evaluation/inference_samd.py \
  --samd_len_threshold 10 \
  --fusion_mode naive \
  ...
```

**方案 2**: 实施 Eagle 优先级保护（方案 B）

#### 情况 C: SKIP < 30%

**说明**: 动态 SAM 在 HumanEval 上匹配质量普遍 >= 5

**原因**: 可能 HumanEval 的 prompt 质量高，动态 SAM 能学到不错的模式

**解决**: 直接跳过质量门控，实施方案 B（Eagle boost + min_ratio）

---

## 📊 Match Quality 分析

如果想详细分析 match quality 分布：

```bash
# 提取所有 match quality
grep '\[GATE\] token=' debug_3.log | \
  sed -n 's/.*best=\([0-9]*\).*/\1/p' | \
  sort -n | uniq -c

# 输出示例：
#   5 2    # 5 次 match=2
#   3 5    # 3 次 match=5
#   8 8    # 8 次 match=8
#   2 12   # 2 次 match=12
```

---

## 🎯 目标

通过这次 debug 测试，明确：
1. 质量门控是否真的触发了
2. SAM match quality 的真实分布
3. 为什么 Eagle accept rate 这么低
4. 下一步应该用哪个方案修复

---

## ⏱️ 预计时间

- Step 1 同步代码: 1 分钟
- Step 2 运行 3 样本: 2-3 分钟
- Step 3 分析结果: 1 分钟
- **总计: < 5 分钟**

请执行后把 `grep "\[GATE\]" debug_3.log` 的输出发给我！
