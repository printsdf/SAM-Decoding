# Phase 3.1: 提取 Eagle3 真实 Logprob

## 目标

替换 naive fusion 中的 depth proxy，使用 Eagle3 真实的 logprob，提升融合质量。

**预期收益**: 5-10% 性能提升（MT-Bench）

---

## 当前问题

### Naive Fusion 的评分策略

```python
# samd/fusion/naive_fusion.py
def parse_eagle_tree(eagle_tree):
    for node in tree_nodes:
        # 问题：使用深度作为分数代理
        node.score = 1.0 / (node.depth + 1)  # ❌ 不准确
```

**问题**：
- 深度不反映 token 的真实接受概率
- 浅层低质量 token 可能得到高分
- 深层高质量 token 可能得到低分
- 导致融合时错误排序

---

## 解决方案

### Step 1: 修改 Eagle3 返回 logprob

**位置**: `samd/tree_model/eagle3/eagle3_model.py`

**当前实现**：
```python
def topK_genrate(self, hidden_states, input_ids, head, max_length=4):
    # ... 生成逻辑 ...
    
    # 当前只返回 tokens 和 buffers
    return draft_tokens, tree_buffers
```

**修改后**：
```python
def topK_genrate(self, hidden_states, input_ids, head, max_length=4):
    # ... 生成逻辑 ...
    
    # 新增：收集每个 token 的 logprob
    logprobs = []
    
    for depth in range(max_length):
        # 前向传播
        logits = self.lm_head(hidden_states)
        probs = torch.softmax(logits, dim=-1)
        
        # 提取 topk
        topk_probs, topk_indices = torch.topk(probs, k=self.top_k)
        topk_logprobs = torch.log(topk_probs)  # 新增
        
        # 收集当前层的 logprobs
        logprobs.extend(topk_logprobs.tolist())
        
        # ... 继续生成下一层 ...
    
    return draft_tokens, tree_buffers, logprobs  # 返回 logprobs
```

### Step 2: 传递 logprob 到融合模块

**位置**: `samd/tree_model/eagle3/eagle3.py`

```python
def gen_draft(self, start_token: int):
    # 调用 topK_genrate
    draft_tokens, tree_buffers, logprobs = self.model.topK_genrate(...)
    
    # 返回时包含 logprobs
    return draft_tokens, tree_buffers, logprobs
```

### Step 3: 在融合中使用 logprob

**位置**: `samd/fusion/naive_fusion.py`

```python
def parse_eagle_tree(eagle_tree, eagle_logprobs):
    """
    eagle_tree: {tokens, tree_mask, position_ids, retrieve_indices}
    eagle_logprobs: List[float]，与 tokens 对齐
    """
    nodes = []
    
    # 从 tree_mask 推导树结构
    tree_spec = TreeSpec.from_eagle3_buffers(...)
    
    for i, token in enumerate(eagle_tree["tokens"]):
        if i == 0:
            continue  # 跳过 root
        
        nodes.append(CandidateNode(
            token=token,
            source="eagle",
            score=eagle_logprobs[i],  # ✅ 使用真实 logprob
            depth=compute_depth(i, tree_spec),
            path=get_path(i, tree_spec),
        ))
    
    return nodes
```

### Step 4: 更新调用链

**位置**: `samd/utils.py::gen_candidates()`

```python
elif samd_config.fusion_mode == "naive":
    start_token = sample_p.squeeze(0).argmax(-1).item()
    
    # ... 质量门控 ...
    
    # 生成 Eagle tree（带 logprob）
    eagle_tokens, eagle_buffers, eagle_logprobs = draft.tree_model.gen_draft(start_token)
    eagle_tree = {
        "tokens": eagle_tokens,
        "logprobs": eagle_logprobs,  # 新增
        **eagle_buffers,
    }
    
    # ... SAM 生成 ...
    
    # 融合（传入 logprobs）
    fused_tree = fuse_eagle_sam_naive(
        eagle_tree=eagle_tree,
        sam_candidates=sam_candidates,
        start_token=start_token,
        config=samd_config.fusion_config,
        sam_match_length=sam_match_length,
    )
```

---

## 技术细节

### Logprob 与 Tree 结构对齐

**问题**: Eagle3 的 tree 是 flattened 的，如何对应 logprob？

**解决**: 
```python
# Eagle3 生成顺序：BFS 逐层生成
# tokens = [root, layer1_tok1, layer1_tok2, ..., layer2_tok1, ...]
# logprobs 与 tokens 一一对应

# 在 parse_eagle_tree 中：
for i, token in enumerate(tokens[1:], start=1):  # 跳过 root
    node.score = logprobs[i]  # 直接索引
```

### Logprob 范围归一化

Eagle logprob 通常在 `[-20, 0]` 范围，SAM match_length 在 `[0, 40]`。

**方案 A**: 分别归一化（当前 naive fusion 做法）
```python
eagle_scores_norm = normalize(eagle_scores)  # -> [0, 1]
sam_scores_norm = normalize(sam_scores)      # -> [0, 1]
```

**方案 B**: 转换为概率后统一
```python
eagle_probs = torch.exp(eagle_logprobs)  # logprob -> prob
# 然后统一归一化
```

**建议**: 先用方案 A（与现有代码一致），Phase 3.2 再用方案 B（payoff 校准）

---

## 实施计划

### P0: 最小修改（验证可行性）

1. **修改 Eagle3 返回**（1 小时）
   - `eagle3_model.py::topK_genrate()`
   - `eagle3.py::gen_draft()`

2. **更新 fusion 模块**（1 小时）
   - `naive_fusion.py::parse_eagle_tree()` 接收 logprobs
   - `naive_fusion.py::fuse_eagle_sam_naive()` 传递 logprobs

3. **更新调用链**（30 分钟）
   - `utils.py::gen_candidates()` 传递 logprobs

4. **编译测试**（10 分钟）
   - 确保无语法错误
   - 单元测试通过

### P1: MT-Bench 验证（2 小时）

```bash
# 在 MT-Bench 上重新评估 naive fusion
python evaluation/inference_samd.py \
  --bench_name mt_bench \
  --fusion_mode naive \
  --answer_file mt_bench_naive_with_logprob.jsonl

# 对比结果
# 之前（depth proxy）: < 130 TPS（预估）
# 现在（real logprob）: > 140 TPS（目标）
```

---

## 验证标准

### 成功标准

- ✅ 编译通过，无运行时错误
- ✅ logprobs 正确传递到融合模块
- ✅ MT-Bench 性能提升 5-10%
- ✅ 分数分布更合理（可视化验证）

### 失败回退

如果性能没提升：
1. 检查 logprob 提取是否正确（打印验证）
2. 检查归一化是否合理（对比分布）
3. 如果仍然失败 → logprob 本身可能不够，直接进入 Phase 3.2（payoff 校准）

---

## 预期输出

### 统计输出示例

```
Naive Fusion Analysis (with real logprob)
============================================================
Average nodes per step:
  Eagle: 42.3
  SAM:   18.7
  
Score distribution:
  Eagle logprob range: [-15.2, -0.3]
  SAM match_length range: [0, 12]
  
Acceptance rates:
  Eagle: 45.2% (vs 13.7% with depth proxy) ✅ 大幅提升
  SAM:   18.3%
  
Total tokens/sec: 142.5 (vs 130 estimated) ✅ 提升 9.6%
============================================================
```

---

## 后续步骤

Phase 3.1 完成后：
- **Phase 3.2**: 实现 Payoff 校准器
- **Phase 3.3**: Context-aware fusion weight
- **Phase 3.4**: 完整实验 + 论文

---

## Codex 执行指令

```bash
codex exec "实现 Phase 3.1：提取 Eagle3 真实 logprob。

参考规格：.trellis/tasks/06-06-dual-draft-fusion/docs/implementation/phase3.1_eagle3_logprob.md

任务：
1. 修改 eagle3_model.py::topK_genrate() 返回 logprobs
2. 修改 eagle3.py::gen_draft() 传递 logprobs
3. 修改 naive_fusion.py::parse_eagle_tree() 使用 logprobs
4. 修改 utils.py::gen_candidates() 传递 logprobs

要求：
- 保持向后兼容（其他 fusion mode 不受影响）
- logprobs 与 tokens 对齐
- 编译测试通过

验证：生成测试命令，准备 MT-Bench 评估" \
  --cd /Users/printsdf/Research/paper_code/SAM-Decoding \
  --sandbox workspace-write
```
