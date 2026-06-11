# Phase 3.1 性能优化方案

## 问题确认

**Experiment 1 结果**：
```
pure_eagle3:        55.917 TPS (baseline)
fusion_100skip:     52.975 TPS (0.947x) ❌ 慢 5.3%
naive_logprob:      53.539 TPS (0.957x)
```

**结论**：融合路径本身有 ~5% 性能开销，这是主要问题。

---

## 性能开销来源

### 1. Debug 日志（最大疑点）

远端代码有 `[GATE]` 逐 token 打印：
```python
print(f"[GATE] token={start_token}, dyn={match_dyn}, ...", flush=True)
```

**影响**：
- 每次生成调用一次（~10000 次）
- `flush=True` 强制刷新 I/O
- 字符串格式化开销

**预期收益**：移除后可能恢复 3-4% 性能

### 2. SAM 质量检查

即使 threshold=999999，仍然调用：
```python
index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
index_static, match_static = draft.sam_static.lookup(start_token)
```

**影响**：
- 每次生成都查询 SAM（即使不用）
- 动态 SAM lookup 可能有小开销

**预期收益**：移除后可能恢复 0.5-1% 性能

### 3. 统计记录

```python
draft.record_naive_fusion({...})  # 每次记录大量字段
```

**影响**：
- 字典构建和拷贝
- 累积统计计算

**预期收益**：优化后可能恢复 0.5% 性能

### 4. 树解析（Eagle-only 分支）

```python
eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
# ... 构建 metadata dict
return Candidates(...)
```

**影响**：
- 额外的字典构建
- Candidates 封装

**预期收益**：优化后可能恢复 0.5% 性能

---

## 优化方案

### Step 1: 移除 Debug 日志（远端）

**远端修改** `samd/utils.py`：

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 备份
cp samd/utils.py samd/utils.py.bak_debug

# 移除所有 [GATE] 打印
sed -i '/print.*\[GATE\]/d' samd/utils.py
sed -i '/print.*flush=True/d' samd/utils.py

# 或手动编辑，注释掉：
# - print(f"[GATE] token=...")
# - print(f"[GATE] -> SKIP...")
# - print(f"[GATE] -> FUSE...")
```

**验证**：
```bash
# 重新运行 Experiment 1
python evaluation/inference_samd.py \
  --fusion_mode naive \
  --samd_len_threshold 999999 \
  --answer_file fusion_100skip_no_debug.jsonl

# 预期：TPS 提升到 54-55（恢复 1-2 TPS）
```

### Step 2: 延迟 SAM lookup（如果 Step 1 不够）

**修改** `samd/utils.py`：

```python
if samd_config.fusion_mode == "naive":
    start_token = sample_p.squeeze(0).argmax(-1).item()

    # 延迟 SAM lookup：只在需要时才查询
    # index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)  # 移到这里之前

    samd_len_threshold = getattr(samd_config, "samd_len_threshold", None)
    if samd_len_threshold is None:
        samd_len_threshold = samd_config.len_threshold

    # 优化：如果 threshold 极高，直接跳过 SAM
    if samd_len_threshold >= 999999:
        # 完全跳过 SAM，直接 Eagle-only
        eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
        # ... 直接返回

    # 正常融合路径才查询 SAM
    index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
    # ...
```

**预期收益**：对 100% skip 场景可能额外恢复 0.5-1% 性能

### Step 3: 轻量化统计（如果需要）

**修改** `samd/draft.py::record_naive_fusion()`：

```python
def record_naive_fusion(self, metadata: Dict) -> None:
    if not self.fusion_stats.get("enabled"):
        return

    # 只记录关键统计，减少字段
    self.fusion_stats["steps"] += 1
    self.fusion_stats["eagle_nodes_sum"] += int(metadata.get("eagle_nodes", 0))
    self.fusion_stats["sam_nodes_sum"] += int(metadata.get("sam_nodes", 0))

    # 跳过详细字段（在非调试模式）
    if not samd_config.debug_mode:
        return

    # 详细统计只在 debug 模式记录
    # ... 其他字段
```

### Step 4: 快速路径优化（Eagle-only fallback）

当 100% skip 时，应该走最快的路径：

```python
# 在 gen_candidates() 中
if samd_config.fusion_mode == "naive" and samd_len_threshold >= 999999:
    # 快速路径：直接等价于 fusion_mode="none"
    candidate_type, tokens, buffers_kwargs = draft.lookup(start_token)
    # 使用原有的 Eagle-only 路径，完全绕过融合逻辑
    # ...
```

---

## 实施优先级

### Priority 1（立即）：移除 debug 日志

**执行**：
```bash
# 远端
cd /teamspace/studios/this_studio/SAM-Decoding

# 方式 1：sed 自动移除
sed -i.bak '/\[GATE\]/d' samd/utils.py

# 方式 2：手动编辑（更安全）
vi samd/utils.py
# 搜索 [GATE]，注释掉所有相关行

# 验证
grep -n "GATE" samd/utils.py  # 应该没有结果

# 重新测试
PYTHONPATH=. python evaluation/inference_samd.py \
  --fusion_mode naive \
  --samd_len_threshold 999999 \
  --max_cache_len 4096 \
  --answer_file fusion_100skip_clean.jsonl
```

**预期**：
- 之前：52.975 TPS
- 优化后：54-55 TPS（恢复 1-2 TPS）

### Priority 2（如果 P1 恢复到 54-55）：分析剩余开销

如果移除 debug 后到达 54-55 TPS：
- 剩余 1-2% 开销可接受
- 继续验证 naive_logprob 是否也提升
- 如果 naive_logprob 到 55+ TPS → 成功！

### Priority 3（如果仍然慢）：深度 profile

```bash
# Python profiler
python -m cProfile -o profile.stats evaluation/inference_samd.py ...

# 分析热点
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

---

## 成功标准

### 优化目标

| Config | Before | Target | Status |
|--------|--------|--------|--------|
| fusion_100skip | 52.975 TPS | > 54.5 TPS | ⏳ 待优化 |
| naive_logprob | 53.539 TPS | > 55.0 TPS | ⏳ 待优化 |

### 判断标准

**✅ 优化成功**：
- fusion_100skip > 54.5 TPS（< 2.5% 慢于 baseline）
- naive_logprob > 55.0 TPS（接近 baseline）
- 融合路径开销 < 2%（可接受）

**⚠️ 部分成功**：
- fusion_100skip 54-54.5 TPS
- 需要进一步优化或接受小开销

**❌ 优化失败**：
- fusion_100skip < 54 TPS
- 需要重新设计融合路径

---

## 时间估算

- Step 1: 30 分钟（移除 debug + 重新测试）
- Step 2-4: 1-2 小时（如果需要）

**总计**：0.5-2.5 小时可以完成优化

---

## 下一步

**立即执行 Priority 1**：

```bash
# 远端
cd /teamspace/studios/this_studio/SAM-Decoding

# 移除 debug 日志
sed -i.bak_gate '/\[GATE\]/d' samd/utils.py

# 验证
grep GATE samd/utils.py  # 应该空

# 重新测试
PYTHONPATH=. python evaluation/inference_samd.py \
  --template llama3 \
  --model-type llama3 \
  --model-path "$MODEL_PATH" \
  --bench-name mt_bench \
  --fusion_mode naive \
  --samd_len_threshold 999999 \
  --max_cache_len 4096 \
  --answer_file evaluation/data/mt_bench/model_answer/fusion_100skip_clean.jsonl \
  2>&1 | tee logs/fusion_100skip_clean.log
```

把结果告诉我！
