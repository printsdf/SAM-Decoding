"""Candidate selectors and MAT for same-trace oracle analysis."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

from evaluation.oracle.types import DecodeStep, OracleCandidate


def candidate_nonroot_path(candidate: OracleCandidate) -> Tuple[int, ...]:
    if candidate.depth <= 0:
        return ()
    if candidate.token_path:
        if len(candidate.token_path) >= candidate.depth + 1:
            return tuple(candidate.token_path[-candidate.depth :])
        return tuple(candidate.token_path)
    if candidate.path:
        parent_nonroot = tuple(candidate.path[1:])
        parent_depth = max(candidate.depth - 1, 0)
        if parent_depth == 0:
            parent_nonroot = ()
        elif len(parent_nonroot) > parent_depth:
            parent_nonroot = parent_nonroot[-parent_depth:]
        return parent_nonroot + (int(candidate.token),)
    return ()


def candidate_matches_acceptance(candidate: OracleCandidate, step: DecodeStep) -> bool:
    if not candidate.accepted:
        return False
    if not step.acceptance_path:
        return True
    nonroot_path = candidate_nonroot_path(candidate)
    if nonroot_path:
        if len(nonroot_path) > len(step.acceptance_path):
            return False
        return list(nonroot_path) == step.acceptance_path[: len(nonroot_path)]
    index = candidate.depth - 1
    return 0 <= index < len(step.acceptance_path) and step.acceptance_path[index] == candidate.token


def _candidate_truth(candidate: OracleCandidate, step: DecodeStep) -> bool:
    return candidate_matches_acceptance(candidate, step)


def _rank_key(step: DecodeStep, candidate: OracleCandidate) -> Tuple[int, int, float, int]:
    truth = 1 if _candidate_truth(candidate, step) else 0
    return (-truth, int(candidate.depth), -float(candidate.score), int(candidate.token))


def _apply_limit(candidates: List[OracleCandidate], limit: Optional[int]) -> List[OracleCandidate]:
    if limit is None:
        return candidates
    return candidates[: max(int(limit), 0)]


def select_perfect(step: DecodeStep, limit: Optional[int]) -> List[OracleCandidate]:
    """Upper-bound selector: true-accepted path candidates first."""
    return _apply_limit(sorted(step.candidates, key=lambda item: _rank_key(step, item)), limit)


def select_budgeted(step: DecodeStep, node_budget: int) -> List[OracleCandidate]:
    """Oracle selector that maximizes accepted-token payoff per selected node."""
    budget = max(int(node_budget), 0)
    ranked = sorted(step.candidates, key=lambda item: _rank_key(step, item))
    return ranked[:budget]


def select_source_balanced(
    step: DecodeStep,
    limit: int,
    min_sam_ratio: float = 0.30,
    max_sam_ratio: float = 0.70,
) -> List[OracleCandidate]:
    """Oracle selector with an approximate 30-70% SAM source-ratio constraint."""
    limit = max(int(limit), 0)
    if limit <= 0:
        return []

    ranked = sorted(step.candidates, key=lambda item: _rank_key(step, item))
    min_sam = int(math.ceil(limit * min_sam_ratio))
    max_sam = int(math.floor(limit * max_sam_ratio))
    selected: List[OracleCandidate] = []
    used = set()

    def choose(source: Optional[str]) -> bool:
        sam_count = sum(1 for item in selected if item.source == "sam")
        for index, candidate in enumerate(ranked):
            if index in used:
                continue
            if source is not None and candidate.source != source:
                continue
            if candidate.source == "sam" and sam_count >= max_sam:
                continue
            used.add(index)
            selected.append(candidate)
            return True
        return False

    while len(selected) < limit:
        sam_count = sum(1 for item in selected if item.source == "sam")
        if sam_count < min_sam and choose("sam"):
            continue
        if choose(None):
            continue
        break
    return selected


def select_eagle3_only(step: DecodeStep, limit: Optional[int]) -> List[OracleCandidate]:
    return _apply_limit([item for item in step.candidates if item.source == "eagle"], limit)


def select_sam_sequence_graft(step: DecodeStep, limit: Optional[int]) -> List[OracleCandidate]:
    eagle = [item for item in step.candidates if item.source == "eagle"]
    sam = [item for item in step.candidates if item.source == "sam"]
    return _apply_limit(eagle + sam, limit)


def selected_mat(step: DecodeStep, selected: Iterable[OracleCandidate]) -> float:
    selected = list(selected)
    # SAMD accept-length metrics count the verifier's root/start token for each
    # decode step. Candidate records are non-root nodes, so add that token here
    # to keep oracle MAT comparable with evaluation accept_lengths.
    root_token = 1.0
    if step.acceptance_path:
        accepted_by_depth: Dict[int, set] = {}
        for candidate in selected:
            if _candidate_truth(candidate, step):
                accepted_by_depth.setdefault(candidate.depth, set()).add(candidate.token)
        accepted = 0
        for depth, token in enumerate(step.acceptance_path, start=1):
            if token not in accepted_by_depth.get(depth, set()):
                break
            accepted += 1
        return root_token + float(accepted)
    return root_token + float(sum(1 for candidate in selected if candidate.accepted))


def default_budget(step: DecodeStep, fallback: Optional[int]) -> int:
    if fallback is not None:
        return int(fallback)
    stats_budget = int(step.stats.get("eagle_nodes", 0)) + int(step.stats.get("sam_nodes", 0))
    return stats_budget if stats_budget > 0 else len(step.candidates)
