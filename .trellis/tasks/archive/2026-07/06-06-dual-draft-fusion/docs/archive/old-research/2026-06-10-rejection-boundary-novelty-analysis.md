# Novelty Analysis: Rejection-Boundary SAM Repair

## Literature Search (2026-06-10)

### Directly Relevant Papers

#### 1. Graft: Draft Less, Retrieve More (arXiv:2605.20104, May 2026)
- **Mechanism**: Prune-then-Graft. Prune low-confidence EAGLE3 branches, graft retrieved tokens into empty slots
- **Difference**: Topology-centric (fill pruned slots) vs our boundary-centric (repair at rejection point)
- **Novelty threat**: High on "fusion" aspect, low on "conditional activation"
- **Key gap**: Graft always grafts where it prunes; doesn't wait for rejection signal

#### 2. D2SD: Dual Diffusion Draft (arXiv:2606.04446, June 2026)
- **Mechanism**: Re-anchoring. Predicts first rejection point, re-anchors second diffusion drafter there
- **Difference**: Neural→Neural repair vs our Neural→Retrieval repair
- **Novelty threat**: HIGH on "rejection boundary" terminology, but uses expensive second neural pass
- **Key gap**: SAM is O(1) and "free"; diffusion adds compute overhead

#### 3. HVD: Hybrid Verified Decoding (arXiv:2606.01019, May 2026, NVIDIA)
- **Mechanism**: Payoff Predictor selects which drafter (Cache vs Model) per step
- **Difference**: Switch (A or B) vs our sequence/repair (A then B at boundary)
- **Novelty threat**: Low. Selection-based, not repair-based

#### 4. RASD: Retrieval-Augmented Speculative Decoding (arXiv:2503.03434, March 2025, Alibaba)
- **Mechanism**: Parallel fusion of neural tree + retrieval tree via Longest Prefix Matching
- **Difference**: Always-on parallel fusion vs our conditional serial activation
- **Novelty threat**: Medium. Covers Neural+Retrieval but lacks boundary triggering

### Comparison Matrix

| Paper | Strategy | Repair Source | Boundary-Aware? | Cost |
|-------|----------|--------------|-----------------|------|
| Graft | Prune-then-fill | Suffix Cache | No (pruning) | Medium |
| D2SD | Re-anchoring | Neural (Diffusion) | **Yes** | High |
| HVD | Selection (switch) | Cache or Model | No (step-level) | Low |
| RASD | Parallel fusion | REST/PLD | No (always-on) | Medium |
| **Ours** | **Boundary repair** | **SAM (O(1))** | **Yes** | **Low** |

### Novelty Assessment

**Our unique contribution has 3 pillars:**

1. **Heterogeneous synergy**: Neural→Retrieval repair (not Neural→Neural like D2SD)
   - SAM is O(1) exact match, zero additional GPU compute
   - Leverages complementary strengths: EAGLE for syntax, SAM for repetition

2. **Conditional activation**: SAM only invoked at predicted rejection boundary
   - Unlike Graft (always fuses) or RASD (always parallel)
   - Saves overhead when EAGLE is confident

3. **Code-specific empirical signal**: +9.65% oracle gap on HumanEval
   - Most papers fight for 2-5% MAT gains over EAGLE3
   - Strong domain-specific result (code boilerplate & structural repetition)

### Positioning Strategy

**Title direction**: "Non-Parametric Boundary Repair for Speculative Decoding"

**Story**:
- D2SD proved rejection-boundary concept works for neural repair
- We show the OPTIMAL repair source at rejection boundaries is not more neural compute, but exact non-parametric retrieval (SAM)
- Zero-cost repair: SAM lookup is O(1), doesn't compete for GPU resources

**Differentiation from closest competitors**:
- vs Graft: We are boundary-targeted, not topology-filling
- vs D2SD: We use free retrieval, not expensive second neural pass
- vs RASD: We are conditional (only at boundary), not always-on fusion
- vs HVD: We repair within a step, not select between steps

### Risk Assessment

**Main risk**: D2SD already uses "rejection boundary" terminology
**Mitigation**: Emphasize that our contribution is the REPAIR SOURCE (retrieval vs neural), not the boundary concept itself. Frame as "what is the optimal repair mechanism at rejection boundaries?"

**Secondary risk**: Graft is close in spirit (neural + retrieval)
**Mitigation**: Graft is topology-first (where to prune determines where to graft). We are confidence-first (boundary prediction drives repair activation). Different causal direction.
