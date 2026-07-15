# Phase 1 Implementation Progress (2026-06-11)

## ✅ Completed (Patches Applied)

### 1. samd/fusion/rejection_boundary.py ✅
Created new module with:
- `compute_confidence_margin(logprobs, depth)` - compute top1-top2 margin
- `should_trigger_sam(margin, threshold)` - decision function

### 2. samd/samd_config.py ✅
Applied changes:
- Added `"rejection_boundary"` to `fusion_mode` Literal type
- Added `rejection_conf_threshold: float = 0.5` config field
- Updated validation to accept "rejection_boundary"
- Modified `fusion_config` property to map rejection_boundary → naive for FusionConfig

### 3. scripts/eval_rejection_boundary_phase1.sh ✅
Created evaluation script for 20 HumanEval samples.

## ⏳ Remaining Work

### 4. samd/utils.py (Patch 2 - Complex, 130+ lines)

**Location**: Insert BEFORE line 278 (`if samd_config.fusion_mode != "none"`)

**Codex provided full implementation** (see Codex output above). The patch includes:
- Draft EAGLE with logprobs
- Extract root confidence margin from position_ids
- Trigger SAM if margin < threshold
- Fuse using existing `fuse_eagle_sam_naive()`
- Handle both triggered and skipped cases with proper metadata

**Status**: NOT YET APPLIED due to context limit concerns

## Next Steps

### Option A: Apply Patch 2 Now
Continue in this session and insert the ~130 line code block into `samd/utils.py`.

**Risk**: Context at 141K/200K tokens. May hit limit.

### Option B: Apply in New Session
1. Save current progress (commit samd_config.py + scripts)
2. Open new Claude session
3. Provide Codex's Patch 2 output
4. Apply remaining patch

**Recommended**: Option B for cleaner context.

### Option C: Manual Application
You apply Codex's Patch 2 manually from the output above.

## Codex Patch 2 Summary

The patch adds an `elif samd_config.fusion_mode == "rejection_boundary":` branch that:
1. Calls `draft.tree_model.gen_draft(start_token, return_logprobs=True)`
2. Extracts root-level logprobs from EAGLE tree
3. Computes confidence margin
4. If margin < threshold:
   - Drafts SAM candidates
   - Calls `fuse_eagle_sam_naive()`
   - Returns fused tree
5. Else:
   - Returns pure EAGLE tree
   - Logs "sam_skipped=True"

**Full patch code is in the Codex output above** (search for "Patch 2/3").

## Testing After Completion

```bash
# On remote GPU environment
bash scripts/eval_rejection_boundary_phase1.sh
```

Expected output:
- SAM trigger rate: 5-20%
- No crashes
- MAT ≥ eagle-only baseline
