# 诊断：为什么质量门控没有触发

## 问题

远端 10 样本测试显示：
- `sam_skipped_count = 0`（门控从未触发）
- `len_threshold = 5`（配置正确）
- 但之前统计 SAM avg match=2.1（应该 < 5）

**矛盾**：为什么 match=2.1 的 SAM 没有被门控拒绝？

## 可能原因

### 假设 1：之前的 2.1 是融合后的平均值，不是门控时的值

```python
# 融合后的统计：sam_avg_match_length = 2.1
# 这是已经融合的 SAM 节点的平均匹配长度

# 但门控时检查的是：best_match = max(match_dyn, match_static)
# 这可能是单次查询的匹配长度，不是平均值
```

### 假设 2：动态 SAM 的 match 可能比静态 SAM 高

```python
best_match = max(match_dyn, match_static - len_bias)

# 如果 match_dyn 经常 >= 5，即使 match_static 很低
# 门控也不会触发
```

### 假设 3：len_bias 的影响

```python
# match_static_adjusted = match_static - draft.len_bias
# 如果 len_bias 是负数或 0
# match_static 不会被降低
```

## 诊断命令

在远端执行，添加详细的门控日志：

### 方案 1：修改代码添加 debug 日志

在 `samd/utils.py` 的质量门控部分添加打印：

```python
if samd_config.fusion_mode == "naive":
    start_token = sample_p.squeeze(0).argmax(-1).item()
    index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
    index_static, match_static_raw = draft.sam_static.lookup(start_token)
    match_static = match_static_raw - draft.len_bias
    best_match = max(match_dyn, match_static)
    
    # 添加 debug 日志
    print(f"[DEBUG] start_token={start_token}, match_dyn={match_dyn}, match_static={match_static}, best_match={best_match}, threshold={samd_len_threshold}")
    
    if best_match < samd_len_threshold:
        print(f"[DEBUG] SAM SKIPPED: best_match={best_match} < threshold={samd_len_threshold}")
        # ... Eagle-only 分支
    else:
        print(f"[DEBUG] SAM FUSION: best_match={best_match} >= threshold={samd_len_threshold}")
        # ... 融合分支
```

然后重新跑 10 样本：

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

python evaluation/inference_samd.py \
  --fusion_mode naive \
  --question_begin 0 --question_end 10 \
  --answer_file ./humaneval_debug_10.jsonl \
  2>&1 | tee debug_10.log

# 查看 debug 信息
grep "\[DEBUG\]" debug_10.log | head -50
```

### 方案 2：分析现有 JSONL metadata

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 如果 JSONL 包含 sam_match_quality 字段
python3 << 'EOF'
import json

with open('humaneval_naive_gated_test_10.jsonl') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        meta = data.get('metadata', {})
        fs = meta.get('fusion_stats', {})
        if fs:
            print(f"Sample {i}:")
            print(f"  sam_skipped: {fs.get('sam_skipped', False)}")
            print(f"  sam_match_quality: {fs.get('sam_match_quality', 'N/A')}")
            print(f"  sam_avg_match_length: {fs.get('sam_avg_match_length', 'N/A')}")
            print(f"  threshold: {fs.get('threshold', 'N/A')}")
            print()
EOF
```

### 方案 3：检查 len_bias 值

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from samd.samd_config import SamdConfig

config = SamdConfig()
print(f"Default len_threshold: {config.len_threshold}")
print(f"Default len_bias: {config.len_bias}")
print(f"Default n_predicts: {config.n_predicts}")
EOF
```

## 预期结果

### 如果门控正常工作

```
[DEBUG] samples with best_match < 5: 7-8 / 10
[DEBUG] SAM SKIPPED count: 7-8
sam_skipped_count: 7-8
```

### 如果门控没触发

```
[DEBUG] best_match 分布: 大多数 >= 5
[DEBUG] SAM FUSION count: 8-9
sam_skipped_count: 0  ← 当前情况
```

**如果是后者**：说明动态 SAM 的实际匹配质量比我们预期的好（match >= 5），质量门控正确地让这些通过了。

## 下一步行动

### 情况 A：门控正确，但 match >= 5 很常见

**说明**: 动态 SAM 在某些 token 上匹配质量不错，问题不在质量门控

**解决**: 需要方案 B（Eagle 优先级保护），即使 SAM 质量"还行"，也要保护 Eagle tree

### 情况 B：门控有 bug，应该触发但没触发

**说明**: 代码逻辑有问题，比如 threshold 比较错误

**解决**: 修复 bug 后重新测试

### 情况 C：threshold=5 太低了

**说明**: match=5-8 的 SAM 仍然质量不够好，但通过了门控

**解决**: 提高 threshold 到 8 或 10，或直接用方案 B

---

## 建议

**优先级 1（今天）**: 运行方案 1 的 debug 版本，看看实际的 best_match 分布

**优先级 2（根据结果）**:
- 如果 best_match 多数 >= 5：实施方案 B（Eagle boost + min_ratio）
- 如果 best_match 多数 < 5 但门控没触发：修 bug
- 如果不确定：提高 threshold 到 10 重新测试
