# Quick Debug Patch for Quality Gating

## 问题

fusion_stats 没有保存到 JSONL，无法从文件看到质量门控是否触发。

## 解决方案：添加 stdout debug 日志

### 修改 `samd/utils.py`

在质量门控部分添加 print 语句（临时 debug）：

```python
# 找到这段代码（约 line 110-155）
if samd_config.fusion_mode == "naive":
    start_token = sample_p.squeeze(0).argmax(-1).item()
    index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
    index_static, match_static_raw = draft.sam_static.lookup(start_token)
    match_static = match_static_raw - draft.len_bias
    best_match = max(match_dyn, match_static)
    samd_len_threshold = getattr(samd_config, "samd_len_threshold", None)
    if samd_len_threshold is None:
        samd_len_threshold = samd_config.len_threshold

    # 添加这行 debug 日志（约 line 120）
    print(f"[GATE] token={start_token}, dyn={match_dyn}, static={match_static}, best={best_match}, threshold={samd_len_threshold}", flush=True)

    if best_match < samd_len_threshold:
        # 添加这行
        print(f"[GATE] -> SKIP SAM (best={best_match} < {samd_len_threshold})", flush=True)

        eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
        # ... 后续代码不变

    # 在 else 分支也添加（约 line 155，fusion 分支开始处）
    print(f"[GATE] -> FUSE SAM (best={best_match} >= {samd_len_threshold})", flush=True)

    eagle_tokens, eagle_buffers = draft.tree_model.gen_draft(start_token)
    # ... 后续代码不变
```

### 快速应用（远端）

**方式 1: 用 sed 快速插入**

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

# 备份
cp samd/utils.py samd/utils.py.bak_debug

# 在 best_match = max(...) 后面插入 debug 行
# 找到行号
grep -n "best_match = max(match_dyn, match_static)" samd/utils.py

# 假设是 line 113，在下一行插入
sed -i '113a\    print(f"[GATE] token={start_token}, dyn={match_dyn}, static={match_static}, best={best_match}, threshold={samd_len_threshold}", flush=True)' samd/utils.py

# 在 if best_match < 分支后插入
grep -n "if best_match < samd_len_threshold:" samd/utils.py
# 假设是 line 115，在下一行插入
sed -i '115a\        print(f"[GATE] -> SKIP SAM (best={best_match} < {samd_len_threshold})", flush=True)' samd/utils.py

# 在 fusion 分支插入（找到 eagle_tokens, eagle_buffers = draft.tree_model.gen_draft 那行的前一行）
grep -n "eagle_tokens, eagle_buffers = draft.tree_model.gen_draft" samd/utils.py
# 假设是 line 157，在前一行插入
sed -i '156a\    print(f"[GATE] -> FUSE SAM (best={best_match} >= {samd_len_threshold})", flush=True)' samd/utils.py
```

**方式 2: 手动编辑**

直接用 vi/nano 编辑 `samd/utils.py`，在上面标注的位置添加 3 行 print。

### 重新运行测试

```bash
cd /teamspace/studios/this_studio/SAM-Decoding

python evaluation/inference_samd.py \
  --fusion_mode naive \
  --question_begin 0 --question_end 3 \
  --answer_file ./humaneval_debug_3.jsonl \
  2>&1 | tee debug_3.log

# 查看门控决策
grep "\[GATE\]" debug_3.log
```

### 预期输出

```
[GATE] token=123, dyn=2, static=1, best=2, threshold=5
[GATE] -> SKIP SAM (best=2 < 5)
[GATE] token=456, dyn=8, static=3, best=8, threshold=5
[GATE] -> FUSE SAM (best=8 >= 5)
[GATE] token=789, dyn=1, static=0, best=1, threshold=5
[GATE] -> SKIP SAM (best=1 < 5)
...

总结：SKIP 次数 vs FUSE 次数
```

## 根据结果决定下一步

### 情况 A: 大部分 SKIP（> 70%）

**说明**: 质量门控正常工作，但 Eagle accept rate 仍然低

**原因**: 可能是我们的 Eagle-only fallback 分支有问题，或者统计计算有误

**下一步**: 检查 Eagle-only 分支的实现

### 情况 B: 大部分 FUSE（< 30% SKIP）

**说明**: 动态 SAM 的 match quality 实际上 >= 5 很常见

**原因**: threshold=5 太低，或者动态 SAM 在这些样本上确实质量不错

**下一步**:
- 提高 threshold 到 10 测试，或
- 直接实施方案 B（Eagle 优先级保护）

### 情况 C: 约 50% SKIP

**说明**: 门控正常，但 SAM 质量参差不齐

**下一步**: 需要方案 B 来保护 Eagle tree
