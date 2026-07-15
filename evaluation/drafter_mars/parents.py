"""Parent-record reconstruction for Drafter-MARS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from evaluation.boundary import eagle_candidates
from evaluation.oracle import DecodeStep, OracleCandidate


@dataclass(frozen=True)
class ParentRecord:
    """A drafter parent considered by the Drafter-MARS gate."""

    tree_index: int
    depth: int
    token_path: Tuple[int, ...]
    z1: Optional[float]
    z2: Optional[float]
    logprob_top1: Optional[float]
    logprob_top2: Optional[float]
    cumulative_path_logprob: Optional[float]
    rank1_token: Optional[int]
    rank2_token: Optional[int]
    is_top_path: bool
    normalized_path_logprob: Optional[float]

    @property
    def ratio(self) -> Optional[float]:
        if self.z1 is None or self.z2 is None:
            return None
        return float(self.z2) / (float(self.z1) + 1e-10)

    @property
    def logprob_delta(self) -> Optional[float]:
        if self.logprob_top1 is None or self.logprob_top2 is None:
            return None
        return float(self.logprob_top1) - float(self.logprob_top2)


def eagle_parent_records(step: DecodeStep) -> List[ParentRecord]:
    """Reconstruct parent records from step candidates.

    A parent is identified by its parent_index; its top-1/top-2 child logits
    come from the children's captured ``parent_top1_logit``/``parent_top2_logit``
    fields (the parent's z1/z2 shared across its children).
    """
    children_by_parent: Dict[int, List[OracleCandidate]] = {}
    for candidate in eagle_candidates(step):
        if candidate.parent_index is None:
            continue
        children_by_parent.setdefault(int(candidate.parent_index), []).append(candidate)

    # The acceptance path lets us mark greedy top-path parents: a parent is on
    # the top path if its token_path (root-stripped) is a prefix of the
    # verifier's accepted path. Candidate path includes the root/start token;
    # acceptance_path does not.
    acceptance_prefixes: set = set()
    prefix: Tuple[int, ...] = ()
    for token in step.acceptance_path:
        prefix = prefix + (int(token),)
        acceptance_prefixes.add(prefix)

    def _root_stripped(path: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(path[1:]) if path else path

    parents: List[ParentRecord] = []
    for parent_index, children in sorted(children_by_parent.items()):
        ranked = sorted(
            children,
            key=lambda c: (
                c.local_logprob is None,
                -(c.local_logprob if c.local_logprob is not None else 0.0),
                int(c.tree_index if c.tree_index is not None else c.depth),
            ),
        )
        rank1 = ranked[0] if ranked else None
        rank2 = ranked[1] if len(ranked) >= 2 else None
        z1 = rank1.parent_top1_logit if rank1 is not None else None
        z2 = rank1.parent_top2_logit if rank1 is not None else None
        logprob_top1 = rank1.local_logprob if rank1 is not None else None
        logprob_top2 = rank2.local_logprob if rank2 is not None else None
        cumulative = rank1.cumulative_path_logprob if rank1 is not None else None
        rank1_token = int(rank1.token) if rank1 is not None else None
        rank2_token = int(rank2.token) if rank2 is not None else None
        token_path = tuple(rank1.path) if rank1 is not None else ()
        depth = int(rank1.depth) if rank1 is not None else 0
        is_top_path = (
            bool(token_path) and _root_stripped(token_path) in acceptance_prefixes
        )
        normalized_path_logprob: Optional[float] = None
        if cumulative is not None and depth > 0:
            normalized_path_logprob = float(cumulative) / float(depth)
        parents.append(
            ParentRecord(
                tree_index=parent_index,
                depth=depth,
                token_path=token_path,
                z1=z1,
                z2=z2,
                logprob_top1=logprob_top1,
                logprob_top2=logprob_top2,
                cumulative_path_logprob=cumulative,
                rank1_token=rank1_token,
                rank2_token=rank2_token,
                is_top_path=is_top_path,
                normalized_path_logprob=normalized_path_logprob,
            )
        )
    return parents
