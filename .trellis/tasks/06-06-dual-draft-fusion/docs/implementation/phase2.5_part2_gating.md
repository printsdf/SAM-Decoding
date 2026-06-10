# Phase 2.5 Part 2: Quality Gating Implementation

## 诊断结果（10 样本）

```
问题确诊：
- SAM avg match length: 2.10 << 5  (目标 > 5)
- SAM contribution: 18.26% < 20%
- Both proposed: 0.93 << 5
- Selected SAM: 41.2% > 40%
- Eagle accept: 13.74% ❌ (应该 60-70%)
- SAM accept: 10.04%
```

**根本原因**：SAM 质量差（match_length=2.1）但占 41% 候选，破坏了 Eagle tree 的有效性。

---

## 解决方案：质量门控

### 核心思路

学习 `sam_sequence_graft` 的成功经验：
- ✅ sam_sequence_graft 有质量门控：`if match < threshold: return eagle_only`
- ❌ naive_fusion 无质量门控：无条件融合

### 实现位置

`samd/utils.py::gen_candidates()` 函数，在 `fusion_mode == "naive"` 分支

### 具体修改

```python
def gen_candidates(
    sample_p: torch.Tensor,
    base_tree_retrieve_indices: torch.Tensor,
    draft: DraftModel,
    samd_config: SamdConfig,
    gen_config: SamdGenerationConfig,
    device: str,
):
    # ... 前面的代码不变 ...
    
    elif samd_config.fusion_mode == "naive":
        start_token = sample_p.squeeze(0).argmax(-1).item()
        
        # ==================== 新增：质量门控 ====================
        # 检查 SAM 匹配质量
        index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
        index_static, match_static = draft.sam_static.lookup(start_token)
        match_static_adjusted = match_static - draft.len_bias
        best_match = max(match_dyn, match_static_adjusted)
        
        # 质量门控：SAM 质量差时退化到 Eagle-only
        if best_match < samd_config.samd_len_threshold:
            # Step 1: 生成 Eagle tree
            eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
            
            # Step 2: 记录统计
            draft.record_naive_fusion({
                "eagle_nodes": len(eagle_pred_ids),
                "sam_nodes": 0,
                "sam_skipped": True,
                "sam_match_quality": int(best_match),
                "threshold": samd_config.samd_len_threshold,
            })
            
            # Step 3: 提取候选 tokens
            retrieve_indices = eagle_buffers["tree_retrieve_indices"]
            if retrieve_indices is not None and len(retrieve_indices) > 0:
                candidate_tokens = torch.tensor(
                    [eagle_pred_ids[i] for i in retrieve_indices.tolist()],
                    device=device
                )
            else:
                candidate_tokens = torch.tensor(eagle_pred_ids, device=device)
            
            # Step 4: 返回 Eagle-only candidates
            return Candidates(
                type=CandidateType.tree,
                tokens=torch.tensor([eagle_pred_ids], device=device),
                candidate_tokens=candidate_tokens.unsqueeze(0),
                buffers_kwargs=eagle_buffers,
            )
        # ==================== 质量门控结束 ====================
        
        # SAM 质量好，继续融合（现有代码不变）
        eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
        eagle_tree = {
            "tokens": torch.tensor(eagle_pred_ids, device=device),
            **eagle_buffers,
        }
        
        # ... 后续融合逻辑保持不变 ...
```

### 注意事项

1. **不要修改现有融合逻辑**：只在前面添加质量门控分支
2. **返回格式一致**：Eagle-only 返回格式与融合返回格式保持一致
3. **统计记录**：记录 `sam_skipped=True` 和 `sam_match_quality`

---

## 预期效果

### 改进前（当前）
```
Eagle accept: 13.74%  ❌
SAM accept: 10.04%
Selected SAM: 41.2%
Mean accept: 5.798
```

### 改进后（预期）
```
Eagle accept: > 50%  ✅ (接近 Eagle-only 的 60-70%)
SAM accept: > 20%   (只在质量好时用 SAM)
Selected SAM: < 30% (大部分情况用 Eagle-only)
Mean accept: > 6.5  (目标 > 0.9x Eagle3-only 的 6.948)
```

---

## 验证步骤

### Step 1: 编译检查
```bash
python3 -m compileall samd/utils.py
```

### Step 2: 10 样本测试
```bash
python evaluation/inference_samd.py \
  --model_type llama \
  --model_path /path/to/llama \
  --tree_model_path /path/to/eagle3 \
  --bench_name humaneval \
  --question_begin 0 --question_end 10 \
  --fusion_mode naive \
  --answer_file test_gating_10.jsonl
```

**检查点**：
- Eagle accept rate > 50%？
- SAM skipped 比例？
- Mean accept 提升了吗？

### Step 3: 完整评估（如果 Step 2 通过）
```bash
python evaluation/eval_naive_fusion.py \
  --answer_dir naive_fusion_gated_20260607
```

**目标**：
- Mean accept > 6.5 (vs 当前 5.915)
- TPS > 54 (vs 当前 49.48)
- Speedup > 0.9x (vs 当前 0.816x)

---

## 回退方案

如果质量门控效果不好：

1. **调整阈值**：
   - 当前 `samd_len_threshold` 可能太高/太低
   - 尝试不同阈值：`--samd_len_threshold 2 / 3 / 5`

2. **叠加方案 B**：
   - 在融合分支添加 Eagle boost
   - `eagle_boost=1.2`, `min_eagle_ratio=0.7`

3. **分析新的统计**：
   - 看 SAM skipped 的比例
   - 看质量门控通过时的 Eagle accept rate

---

## 实现文件

**修改文件**：
- `samd/utils.py` (gen_candidates 函数)

**不需要修改**：
- `samd/fusion/naive_fusion.py` (融合逻辑保持不变)
- `samd/draft.py` (统计记录已支持 sam_skipped)
- `samd/samd_model.py` (统计输出已支持)

---

## Codex 执行命令

```bash
# 新开 codex session
codex exec "根据 .trellis/tasks/06-06-dual-draft-fusion/docs/implementation/phase2.5_part2_gating.md 实现质量门控。
修改 samd/utils.py 的 gen_candidates 函数，在 fusion_mode=='naive' 分支添加 SAM 质量检查。
如果 best_match < threshold，返回 Eagle-only candidates。
验证：编译检查 + 生成测试命令。" \
  --cd /Users/printsdf/Research/paper_code/SAM-Decoding \
  --sandbox workspace-write
```
