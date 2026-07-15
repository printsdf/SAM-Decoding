"""Candidate accessors for boundary predictors."""

from __future__ import annotations

from typing import List, Optional, Tuple

from evaluation.oracle import DecodeStep, OracleCandidate, candidate_nonroot_path


def question_index(step: DecodeStep) -> Optional[int]:
    for key in ("question_index", "question_idx", "question_number"):
        value = step.metadata.get(key)
        if value is not None:
            return int(value)
    return None


def eagle_candidates(step: DecodeStep) -> List[OracleCandidate]:
    return sorted(
        [candidate for candidate in step.candidates if candidate.source == "eagle"],
        key=lambda item: (
            item.tree_index is None,
            item.tree_index if item.tree_index is not None else item.depth,
            item.depth,
            item.token,
        ),
    )


def sam_candidates(step: DecodeStep) -> List[OracleCandidate]:
    return [candidate for candidate in step.candidates if candidate.source == "sam"]


def eligible_margin_candidates(step: DecodeStep) -> List[OracleCandidate]:
    return [
        candidate
        for candidate in eagle_candidates(step)
        if candidate.sibling_margin is not None
    ]


def candidate_parent_prefix(candidate: OracleCandidate) -> Tuple[int, ...]:
    path = candidate_nonroot_path(candidate)
    return tuple(path[:-1])


def true_parent_prefix(step: DecodeStep) -> Optional[Tuple[int, ...]]:
    if step.first_rejected_depth is None:
        return None
    return tuple(step.acceptance_path[: max(int(step.first_rejected_depth) - 1, 0)])
