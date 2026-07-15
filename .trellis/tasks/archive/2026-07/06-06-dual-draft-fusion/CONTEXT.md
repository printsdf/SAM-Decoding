# Dual Draft Fusion

Research context for EAGLE3 + SAM dual-draft speculative decoding. This glossary
fixes the language used across the task's specs, plans, and results so that
"drafter margin", "rejection boundary", and "oracle" always mean the same thing.

## Verification & Drafting

**MARS**:
A verifier-side margin-aware acceptance rule. At each verified position the
target model's top-2 / top-1 raw logit ratio is computed; when it exceeds a
threshold the verifier relaxes acceptance (accepts a draft token that falls in
the target's top-2). MARS uses target logits and acts during verification.
_Avoid_: drafter margin, adaptive draft threshold

**Drafter-MARS**:
An offline proxy that borrows MARS's top-2 / top-1 raw-logit ratio signal but
computes it on the EAGLE3 drafter's logits (not the target's) and uses it as a
gate for inserting SAM repair candidates (not as a relaxed-acceptance rule). It
is not MARS; it reuses one signal under a different mechanism.
_Avoid_: MARS, drafter-side MARS, adaptive margin predictor

**SAM rescue**:
A SAM draft candidate that matches the target-accepted sequence at a point where
EAGLE3 was rejected, recovering acceptance the drafter lost.
_Avoid_: SAM repair (when distinguishing outcome from mechanism)

## Oracle Accounting

**MAT (Mean Accepted Tokens)**:
Average tokens accepted per decode step. The primary metric; every oracle MAT in
this task is an offline same-trace replay, not wall-clock throughput.
_Avoid_: tokens per second, throughput

**Rejection boundary**:
The EAGLE3 tree parent (equivalently depth) at which the target verifier first
rejects a draft token in a decode step.
_Avoid_: reject point, fail depth

**Same-trace oracle**:
An offline MAT computed by replaying one captured fusion-profile trace under a
fixed node budget, with no online SAM-generation or verification cost. All MAT
comparisons here are same-trace.
_Avoid_: online MAT, measured throughput

**Rejection-boundary oracle**:
The same-trace oracle that inserts SAM only at the true rejection boundary. Its
MAT gap over EAGLE3-only is the ceiling any boundary-gated SAM method can reach.
_Avoid_: perfect oracle, SAM oracle

**Perfect oracle**:
The same-trace oracle with unconstrained SAM insertion; the absolute upper bound
on MAT for the captured trace. Always >= rejection-boundary oracle.
_Avoid_: upper oracle, max oracle

**Node budget**:
The maximum candidate count per decode step (60), held constant across
EAGLE-only, predicted, and oracle MAT so the comparison is same-budget.
_Avoid_: draft budget, token limit

**Recovered oracle ceiling**:
(predicted_MAT - eagle_MAT) / (rejection_boundary_oracle_MAT - eagle_MAT),
computed same-trace and same-split. Never divide one split's gain by another
split's ceiling.
_Avoid_: oracle recovery rate, ceiling fraction

## Predictors & Metrics

**sibling_margin**:
The drafter's top-1 minus top-2 raw logit at a parent. Equals
logprob_top1 - logprob_top2 (logsumexp cancels), so it is already available from
captured logprobs without raw logits. Distinct from the Drafter-MARS ratio.
_Avoid_: margin, logit gap

**Trigger rate**:
Fraction of decode steps on which a predictor fires (inserts / would insert SAM).
_Avoid_: activation rate, fire rate

**Boundary precision**:
Of steps where the predictor fires, the fraction whose predicted boundary is the
true rejection boundary.
_Avoid_: prediction accuracy

**Boundary recall**:
Of steps with a true rejection boundary, the fraction the predictor catches.
_Avoid_: sensitivity
