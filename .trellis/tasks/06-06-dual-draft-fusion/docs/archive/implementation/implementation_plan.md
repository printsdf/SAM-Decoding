# Implementation Plan: Rejection-Boundary SAM Repair

Based on Gemini + Codex review of experiment design.

## Critical Feedback Summary

### Gemini's Key Risks
1. **Serial overhead**: $T_{draft} = T_{eagle} + P(trigger) \cdot T_{sam}$
2. **False positives**: Low confidence ≠ rejection; may trigger SAM in high-entropy but correct regions
3. **Tree slot dilution**: SAM insertion may prune high-prob EAGLE branches
4. **Double-blind failure**: If base model is also uncertain, SAM may add noise

### Gemini's Recommendations
- Use **relative confidence (margin)** instead of absolute logprob: `logprob_top1 - logprob_top2 < threshold_margin`
- Profile SAM retrieval latency distribution (especially tail latency)
- Measure F1-score of rejection prediction (not just correlation)
- Start conservative (low trigger rate) and increase until TPS hits -5% floor

### Codex's Codebase Audit

**EAGLE confidence extraction**:
- File: `samd/tree_model/eagle3/eagle3_model.py::Eagle3Model.topK_genrate()`
- Already returns `draft_logprobs` aligned with `draft_tokens`
- Recommendation: Add per-node `logprob`, `rank`, `margin` fields computed before pruning

**SAM runtime drafting**:
- Files: `samd/sam/dyn_sam.py::lookup()`, `transfer_state()`, `gen_draft_raw()`
- For arbitrary-depth repair: use `transfer_state()` along EAGLE prefix path
- Warning: `gen_draft_raw()` may climb to ancestor via `to_anc()`—log fallback occurrence

**Tree merge**:
- Files: `samd/tree_model/fusion.py::TreeSpec`, `samd/fusion/naive_fusion.py::build_tree_buffers()`
- Verifier (`samd/samd_model.py`) only needs buffers, no rewrite needed
- **New module**: `samd/fusion/rejection_boundary.py`

### Codex's Main Footgun

Oracle rejection = **coverage failure** (no EAGLE branch matches target).
Low-confidence node ≠ coverage failure.
**Risk**: Trigger on wrong prefix → SAM correct elsewhere but useless here.

---

## Implementation Strategy

### Phase 1: 1-Day Prototype (Smoke Test)

**Goal**: Validate confidence gating and overhead, skip arbitrary-depth complexity.

**Approach**: Root-level gating only
```python
# In samd/utils.py::gen_candidates()
if fusion_mode == "rejection_boundary":
    eagle_tree = draft_eagle(return_logprobs=True)
    eagle_root_confidence = compute_margin(eagle_tree.logprobs[root])

    if eagle_root_confidence < THRESHOLD:
        # Trigger SAM at root (existing naive fusion path)
        sam_candidates = draft_sam_from_root()
        tree = fuse_eagle_sam_naive(eagle_tree, sam_candidates)
    else:
        # Skip SAM, pure EAGLE
        tree = eagle_tree
```

**Metrics to log**:
- SAM trigger rate (target: 5-10%)
- Overhead per step (target: < 1ms)
- MAT on 20 samples vs eagle-only

**Pass criteria**: Trigger rate 5-20%, no crashes, MAT > eagle-only

---

### Phase 2: Full Rejection-Boundary (3 days)

After Phase 1 validates gating works, implement full arbitrary-depth repair.

**New files**:

#### 1. `samd/fusion/rejection_boundary.py`

```python
@dataclass
class BoundaryAnchor:
    depth: int
    prefix_tokens: List[int]
    confidence: float
    margin: float  # top1_logprob - top2_logprob

def select_boundary_anchors(
    eagle_tree: EagleTree,
    eagle_scores: Dict,
    threshold: float,
    max_anchors: int = 1
) -> List[BoundaryAnchor]:
    """Select low-confidence EAGLE nodes as SAM repair anchors."""
    # Compute margin at each depth
    # Select anchors where margin < threshold
    # Return top-k by depth (prefer earlier failures)
    pass

def draft_sam_from_prefix(
    sam_draft: DynSAM,
    prefix_tokens: List[int],
    max_len: int = 16,
    min_match: int = 5
) -> Optional[SAMDraft]:
    """Draft SAM continuation from a specific prefix.

    Uses transfer_state() to navigate to prefix, then gen_draft_raw().
    Returns None if match_length < min_match (quality gate).
    """
    # sam_draft.transfer_state(prefix_tokens)
    # candidates = sam_draft.gen_draft_raw(max_len)
    # Filter by min_match
    pass

def merge_at_boundary(
    eagle_tree: TreeSpec,
    anchors: List[BoundaryAnchor],
    sam_drafts: List[SAMDraft],
    max_added_nodes: int = 16,
    keep_eagle: bool = True
) -> TreeSpec:
    """Insert SAM branches at anchor points in EAGLE tree.

    Strategy:
    - Keep full EAGLE tree (no pruning initially)
    - Add SAM nodes as children of anchor nodes
    - Ensure prefix-closed property maintained
    - Cap total added nodes to max_added_nodes
    """
    # Build combined TreeSpec
    # Use TreeSpec.to_buffers() for verifier
    pass
```

#### 2. Update `samd/samd_config.py`

```python
@dataclass
class SamdConfig:
    # ... existing fields ...

    # Rejection-boundary config
    rejection_conf_threshold: float = 0.5
    rejection_signal: str = "margin"  # "margin" | "logprob" | "entropy"
    rejection_max_anchors: int = 1
    rejection_max_added_nodes: int = 16
    rejection_min_sam_match: int = 5
```

#### 3. Update `samd/utils.py::gen_candidates()`

```python
if samd_config.fusion_mode == "rejection_boundary":
    # Draft EAGLE with logprobs
    eagle_tree = draft_eagle(return_logprobs=True)

    # Select boundary anchors
    anchors = select_boundary_anchors(
        eagle_tree,
        eagle_scores,
        threshold=samd_config.rejection_conf_threshold,
        max_anchors=samd_config.rejection_max_anchors
    )

    if not anchors:
        # No low-confidence nodes, skip SAM
        return eagle_tree

    # Draft SAM from each anchor
    sam_drafts = []
    for anchor in anchors:
        sam_draft = draft_sam_from_prefix(
            sam_drafter,
            anchor.prefix_tokens,
            max_len=samd_config.rejection_max_added_nodes,
            min_match=samd_config.rejection_min_sam_match
        )
        if sam_draft:
            sam_drafts.append(sam_draft)

    # Merge SAM branches into EAGLE tree
    if sam_drafts:
        fused_tree = merge_at_boundary(
            eagle_tree,
            anchors,
            sam_drafts,
            max_added_nodes=samd_config.rejection_max_added_nodes
        )
        return fused_tree
    else:
        return eagle_tree
```

---

## Evaluation Plan

### Checkpoint 1: Confidence Signal Validation (1 hour)
```bash
python evaluation/validate_confidence_signal.py \
  --samples 10 \
  --output confidence_correlation.json
```

**Metrics**: Spearman correlation (confidence vs actual rejection depth)
**Pass**: correlation > 0.5

### Checkpoint 2: Phase 1 Smoke Test (4 hours)
```bash
python -m evaluation.inference_samd \
  --bench-name humaneval \
  --question-begin 0 \
  --question-end 20 \
  --fusion_mode rejection_boundary \
  --rejection_conf_threshold 0.5
```

**Pass**: MAT > eagle-only, trigger rate 5-20%

### Checkpoint 3: Threshold Tuning (4 hours)
Grid search on q0-50:
- Thresholds: [0.2, 0.3, 0.4, 0.5, 0.6]
- Signal: margin (recommended) vs logprob

**Select**: Best MAT with TPS ≥ 95% baseline

### Phase 2 Full Eval (8 hours)
```bash
bash scripts/eval_rejection_boundary_humaneval.sh
```

Run on full HumanEval q0-164 with best threshold.

**Success**: MAT > 3.85 (+5%), TPS ≥ 95% baseline

---

## Risk Mitigation

| Risk | Detection | Mitigation |
|------|-----------|------------|
| Confidence doesn't predict rejection | Checkpoint 1 correlation < 0.3 | Fallback to fixed-depth (always trigger at depth=3) |
| SAM overhead too high | TPS drops > 5% | Reduce max_added_nodes, increase min_sam_match |
| False positive rate high | Trigger rate > 20%, but MAT gain < 2% | Increase threshold, use margin instead of logprob |
| Index misses (OOD tasks) | SAM match_length < 5 frequently | Add match quality gate, log OOD rate |

---

## Next Steps

1. ✅ Experiment design approved
2. **Now**: Let Codex implement Phase 1 (1-day prototype)
3. Run Checkpoint 2 on 20 samples
4. If successful, proceed to Phase 2 (full implementation)

Ready to proceed with implementation?
