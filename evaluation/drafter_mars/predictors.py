"""Drafter-MARS predictor variants."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from evaluation.boundary import Prediction, eagle_candidates
from evaluation.drafter_mars.parents import ParentRecord, eagle_parent_records
from evaluation.oracle import DecodeStep, OracleCandidate


def earliest_parent(
    parents: Sequence[ParentRecord],
    predicate: Callable[[ParentRecord], bool],
) -> Optional[ParentRecord]:
    for parent in sorted(parents, key=lambda p: (p.depth, p.tree_index)):
        if predicate(parent):
            return parent
    return None


def prediction_from_parent(
    method: str,
    parent: Optional[ParentRecord],
    step: DecodeStep,
    threshold: float,
) -> Prediction:
    if parent is None:
        return Prediction(method=method, candidate=None, depth=None, threshold=float(threshold))
    match: Optional[OracleCandidate] = None
    for candidate in eagle_candidates(step):
        if (
            candidate.parent_index is not None
            and int(candidate.parent_index) == int(parent.tree_index)
            and candidate.local_logprob is not None
            and parent.logprob_top1 is not None
            and abs(float(candidate.local_logprob) - float(parent.logprob_top1)) < 1e-9
        ):
            match = candidate
            break
    return Prediction(
        method=method,
        candidate=match,
        depth=int(parent.depth),
        score=parent.ratio,
        threshold=float(threshold),
    )


def predict_draft_mars_top_path(step: DecodeStep, theta: float) -> Prediction:
    parents = eagle_parent_records(step)
    target = earliest_parent(
        [p for p in parents if p.is_top_path],
        lambda p: p.ratio is not None and float(p.ratio) > float(theta),
    )
    return prediction_from_parent("draft_mars_top_path", target, step, theta)


def predict_draft_delta_top_path(step: DecodeStep, tau: float) -> Prediction:
    parents = eagle_parent_records(step)
    target = earliest_parent(
        [p for p in parents if p.is_top_path],
        lambda p: p.logprob_delta is not None and float(p.logprob_delta) < float(tau),
    )
    return prediction_from_parent("draft_delta_top_path", target, step, tau)


def reachable_threshold(
    parents: Sequence[ParentRecord],
    quantile: float,
) -> Optional[float]:
    values = sorted(
        p.normalized_path_logprob
        for p in parents
        if p.normalized_path_logprob is not None
    )
    if not values:
        return None
    quantile = max(0.0, min(1.0, float(quantile)))
    index = int(round(quantile * (len(values) - 1)))
    return values[index]


def predict_draft_mars_reachable(
    step: DecodeStep,
    theta: float,
    reachable_quantile: float,
) -> Prediction:
    parents = eagle_parent_records(step)
    threshold = reachable_threshold(parents, reachable_quantile)
    if threshold is None:
        return Prediction(
            method="draft_mars_reachable",
            candidate=None,
            depth=None,
            threshold=float(theta),
        )

    def predicate(p: ParentRecord) -> bool:
        if p.normalized_path_logprob is None or p.ratio is None:
            return False
        return float(p.normalized_path_logprob) >= float(threshold) and float(p.ratio) > float(theta)

    target = earliest_parent(parents, predicate)
    return prediction_from_parent("draft_mars_reachable", target, step, theta)
