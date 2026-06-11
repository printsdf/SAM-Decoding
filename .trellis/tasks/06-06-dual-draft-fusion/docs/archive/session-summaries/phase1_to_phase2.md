# Session Summary: Phase 1 Complete + Phase 2 Design Ready

**Date**: 2026-06-11
**Context**: 170K/200K tokens used
**Status**: Ready to implement Phase 2 in new session

---

## Completed Work

### Phase A: Oracle Analysis ✅
- MedQA: +1.45% (reject)
- MT-Bench: +0.61% depth-decoupled, +3.01% rejection-boundary (reject, ceiling too low)
- **HumanEval: +9.65% rejection-boundary** ✅ PASS

### Phase A+: Research Validation ✅
- Codex + Gemini direction exploration
- Novelty analysis (vs D2SD/Graft/RASD)
- Expert reviews identified key risks

### Phase 1: Root-Level Gating (FAILED) ⚠️
- **Implemented**: 8 files, ~300 lines
- **Result**: MAT 6.73 vs baseline 7.58 (-11.2%)
- **Root cause**:
  1. Root-level competition, not boundary relay
  2. SAM pollutes EAGLE budget
  3. 实现 ≠ Oracle 测试的东西

### Phase 2: Design Complete ✅
- **Document**: `.trellis/tasks/06-06-dual-draft-fusion/docs/experiments/specs/2026-06-11-phase2-boundary-graft.md`
- **Approach**: Predicted-boundary SAM grafting (true relay)
- **Key constraint**: Single verification pass (verify is 64% of time)

---

## Phase 2 Implementation Plan

### Algorithm
```
1. EAGLE draft + logprobs
2. Predict rejection depth from per-depth margins
3. Extract EAGLE prefix to predicted depth
4. SAM continuation from that prefix (transfer_state)
5. Graft SAM tail onto EAGLE leaves
6. Single verify of fused tree
```

### Files to Create/Modify
1. **New**: `samd/fusion/boundary_graft.py` (~200 lines)
   - `predict_rejection_depth(logprobs, position_ids, threshold)`
   - `graft_sam_at_depth(eagle_tree, sam_candidates, depth, max_nodes)`

2. **Update**: `samd/utils.py` - add `fusion_mode="boundary_graft"` branch

3. **Update**: `samd/samd_config.py` - add config fields

### Success Criteria
- MAT > 7.96 (+5% over baseline 7.58)
- Oracle ceiling: 8.31 (+9.65%)

---

## Next Session Actions

1. **Codex implements** Phase 2 (boundary_graft)
2. **Gemini + You review** implementation
3. **Run threshold sweep** (6 configs, ~36 min)
4. **Decision**:
   - Success → MT-Bench validation → paper
   - Partial → tune + add gates
   - Failure → negative result or pivot

---

## Key Learnings

### Why Phase 1 Failed
- **Root competition** ≠ **boundary relay** (oracle measured relay)
- SAM from root competes for budget, doesn't utilize EAGLE prefix
- Confidence inversion (high margin = confidently wrong) was fixable but insufficient

### Why Phase 2 Should Work
- Grafts SAM at EAGLE's predicted failure point (true relay)
- Preserves EAGLE's good prefix
- Single verify (no overhead)
- Matches what oracle actually tested

### Critical Implementation Details
- Use `sam.transfer_state()` to walk EAGLE prefix
- Graft only at leaf nodes of predicted depth
- Keep EAGLE full tree, add SAM tail (4-8 nodes)
- Log predicted vs actual rejection depth for analysis

---

## Files Modified (Phase 1)

**New**:
1. `samd/fusion/rejection_boundary.py`
2. `scripts/profile_fusion_humaneval.sh`
3. `scripts/eval_rejection_boundary_phase1.sh`
4. `scripts/eval_eagle_only_humaneval_full.sh`
5. `scripts/eval_rejection_boundary_humaneval_full.sh`
6. `evaluation/oracle_rejection_boundary.py`
7. `evaluation/oracle_high_precision_sam.py`

**Modified**:
1. `samd/samd_config.py` - rejection_boundary config
2. `samd/utils.py` - rejection_boundary mode (157 lines)
3. `evaluation/inference_samd.py` - CLI args

---

## Recommended Next Session Prompt

```
继续 dual-draft-fusion 任务。Phase 1 (root-level) 失败了（MAT下降11%），已设计好 Phase 2 (boundary-graft)。

请：
1. 让 Codex 实现 Phase 2（基于 .trellis/tasks/06-06-dual-draft-fusion/docs/experiments/specs/2026-06-11-phase2-boundary-graft.md）
2. 你和 Gemini review 代码
3. 运行 threshold sweep 测试

Phase 2 核心：在预测的拒绝深度 graft SAM tail，单次 verify，真正的接力而非竞争。

参考文档：
- .trellis/tasks/06-06-dual-draft-fusion/IMPLEMENTATION_COMPLETE.md
- .trellis/tasks/06-06-dual-draft-fusion/docs/experiments/specs/2026-06-11-phase2-boundary-graft.md
```
