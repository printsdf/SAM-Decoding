"""Prediction metrics for boundary predictors."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation.boundary.candidates import (
    candidate_parent_prefix,
    eagle_candidates,
    sam_candidates,
    true_parent_prefix,
)
from evaluation.boundary.types import DEFAULT_NODE_BUDGET, Prediction
from evaluation.oracle import (
    DecodeStep,
    OracleCandidate,
    candidate_matches_acceptance,
    select_eagle3_only,
    select_perfect,
    selected_mat,
)


def is_depth_hit(step: DecodeStep, prediction: Prediction) -> bool:
    return (
        prediction.depth is not None
        and step.first_rejected_depth is not None
        and int(prediction.depth) == int(step.first_rejected_depth)
    )


def is_prefix_hit(step: DecodeStep, prediction: Prediction) -> bool:
    if prediction.candidate is None or not is_depth_hit(step, prediction):
        return False
    true_parent = true_parent_prefix(step)
    return true_parent is not None and candidate_parent_prefix(prediction.candidate) == true_parent


def sam_can_rescue(step: DecodeStep) -> bool:
    if step.first_rejected_depth is None:
        return False
    reject_depth = int(step.first_rejected_depth)
    return any(
        candidate.depth >= reject_depth and candidate_matches_acceptance(candidate, step)
        for candidate in sam_candidates(step)
    )


def oracle_rank_key(step: DecodeStep, candidate: OracleCandidate) -> Tuple[int, int, float, int]:
    truth = 1 if candidate_matches_acceptance(candidate, step) else 0
    return (-truth, int(candidate.depth), -float(candidate.score), int(candidate.token))


def select_oracle_allowed(
    step: DecodeStep,
    candidates: Sequence[OracleCandidate],
    node_budget: Optional[int],
) -> List[OracleCandidate]:
    selected = sorted(candidates, key=lambda candidate: oracle_rank_key(step, candidate))
    if node_budget is None:
        return selected
    return selected[: max(int(node_budget), 0)]


def prediction_mat(
    step: DecodeStep,
    prediction: Prediction,
    node_budget: Optional[int],
) -> float:
    allowed = list(select_eagle3_only(step, None))
    if is_prefix_hit(step, prediction) and sam_can_rescue(step):
        allowed.extend(sam_candidates(step))
    selected = select_oracle_allowed(step, allowed, node_budget)
    return selected_mat(step, selected)


def evaluate_predictions(
    steps: Sequence[DecodeStep],
    predictions: Sequence[Prediction],
    method: str,
    threshold: Optional[float] = None,
    node_budget: Optional[int] = DEFAULT_NODE_BUDGET,
) -> Dict[str, Any]:
    if len(steps) != len(predictions):
        raise ValueError("steps and predictions length mismatch")
    total = len(steps)
    true_rejections = sum(1 for step in steps if step.first_rejected_depth is not None)
    triggered = sum(1 for prediction in predictions if prediction.depth is not None)
    depth_hits = sum(
        1 for step, prediction in zip(steps, predictions)
        if is_depth_hit(step, prediction)
    )
    prefix_hits = sum(
        1 for step, prediction in zip(steps, predictions)
        if is_prefix_hit(step, prediction)
    )
    rescued_hits = sum(
        1 for step, prediction in zip(steps, predictions)
        if is_prefix_hit(step, prediction) and sam_can_rescue(step)
    )
    depth_errors = [
        abs(int(prediction.depth) - int(step.first_rejected_depth))
        for step, prediction in zip(steps, predictions)
        if prediction.depth is not None and step.first_rejected_depth is not None
    ]
    eagle_mat = sum(
        selected_mat(step, select_eagle3_only(step, node_budget))
        for step in steps
    ) / float(total)
    perfect_mat = sum(
        selected_mat(step, select_perfect(step, node_budget))
        for step in steps
    ) / float(total)
    predicted_mat = sum(
        prediction_mat(step, prediction, node_budget)
        for step, prediction in zip(steps, predictions)
    ) / float(total)
    base_rate = true_rejections / float(total) if total else 0.0
    precision = prefix_hits / float(triggered) if triggered else 0.0
    recall = prefix_hits / float(true_rejections) if true_rejections else 0.0
    ceiling = perfect_mat - eagle_mat
    return {
        "method": method,
        "threshold": threshold,
        "node_budget": node_budget,
        "steps": total,
        "true_rejection_steps": true_rejections,
        "triggered_steps": triggered,
        "trigger_rate": triggered / float(total) if total else 0.0,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "precision_lift": precision / base_rate if base_rate > 0 else None,
        "depth_hit_rate": depth_hits / float(triggered) if triggered else 0.0,
        "path_prefix_hit_rate": prefix_hits / float(depth_hits) if depth_hits else 0.0,
        "sam_rescue_rate_on_prefix_hits": rescued_hits / float(prefix_hits) if prefix_hits else 0.0,
        "median_depth_error": statistics.median(depth_errors) if depth_errors else None,
        "eagle_mat": eagle_mat,
        "perfect_mat": perfect_mat,
        "predicted_boundary_oracle_mat": predicted_mat,
        "oracle_gap_vs_eagle": (
            (predicted_mat - eagle_mat) / eagle_mat if eagle_mat > 0 else None
        ),
        "recovered_oracle_ceiling": (
            (predicted_mat - eagle_mat) / ceiling if ceiling > 0 else None
        ),
    }
