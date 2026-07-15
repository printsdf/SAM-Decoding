"""Pure Drafter-MARS gate functions (no model dependency)."""

from typing import List, NamedTuple, Optional, Sequence, Tuple

RATIO_EPS = 1e-10


class TriggerInfo(NamedTuple):
    """Earliest triggering top-path parent."""

    parent_index: int
    parent_depth: int
    child_index: int
    ratio: float


def greedy_top_path(
    parents: Sequence[int],
    logprobs: Optional[Sequence[float]],
) -> List[int]:
    """Node indices on the greedy max-local-logprob descent from the root."""
    n = len(parents)
    if n == 0 or logprobs is None or len(logprobs) != n:
        return []
    children: List[List[int]] = [[] for _ in range(n)]
    for index in range(1, n):
        children[parents[index]].append(index)
    path: List[int] = [0]
    node = 0
    while children[node]:
        node = max(children[node], key=lambda child: logprobs[child])
        path.append(node)
    return path


def greedy_top_path_parents(
    parents: Sequence[int],
    logprobs: Optional[Sequence[float]],
) -> List[int]:
    """Node indices acting as parents on the greedy max-local-logprob root descent."""
    path = greedy_top_path(parents, logprobs)
    return path[:-1]


def top_path_ratio_trigger(
    parents: Sequence[int],
    logprobs: Optional[Sequence[float]],
    raw_pairs: Optional[Sequence[Optional[Tuple[Optional[float], Optional[float]]]]],
    theta: float,
) -> Tuple[bool, Optional[float]]:
    """Max z2/(z1+eps) ratio over top-path parents; triggered iff max ratio > theta.

    ``raw_pairs[i]`` carries the (z1, z2) of the parent that expanded node ``i``
    (capture contract; the root carries ``(None, None)``). Each top-path parent's
    pair is therefore read at its greedy child, i.e. at every non-root path node.
    """
    if raw_pairs is None or len(raw_pairs) != len(parents):
        return False, None
    max_ratio: Optional[float] = None
    for node_index in greedy_top_path(parents, logprobs)[1:]:
        pair = raw_pairs[node_index]
        if pair is None:
            continue
        z1, z2 = pair
        if z1 is None or z2 is None:
            continue
        ratio = float(z2) / (float(z1) + RATIO_EPS)
        if max_ratio is None or ratio > max_ratio:
            max_ratio = ratio
    if max_ratio is None:
        return False, None
    return max_ratio > theta, max_ratio


def all_top_path_triggers(
    parents: Sequence[int],
    logprobs: Optional[Sequence[float]],
    raw_pairs: Optional[Sequence[Optional[Tuple[Optional[float], Optional[float]]]]],
    theta: float,
) -> List[TriggerInfo]:
    """Every triggering top-path parent, in descent (ascending depth) order.

    Same pair-at-greedy-child read as ``earliest_top_path_trigger``; when
    non-empty, the first element equals its result.
    """
    if raw_pairs is None or len(raw_pairs) != len(parents):
        return []
    triggers: List[TriggerInfo] = []
    for depth, child_index in enumerate(greedy_top_path(parents, logprobs)[1:]):
        pair = raw_pairs[child_index]
        if pair is None:
            continue
        z1, z2 = pair
        if z1 is None or z2 is None:
            continue
        ratio = float(z2) / (float(z1) + RATIO_EPS)
        if ratio > theta:
            triggers.append(TriggerInfo(parents[child_index], depth, child_index, ratio))
    return triggers


def earliest_top_path_trigger(
    parents: Sequence[int],
    logprobs: Optional[Sequence[float]],
    raw_pairs: Optional[Sequence[Optional[Tuple[Optional[float], Optional[float]]]]],
    theta: float,
) -> Optional[TriggerInfo]:
    """Earliest (shallowest; tie-break lowest tree index) triggering top-path parent.

    ``raw_pairs[i]`` carries the (z1, z2) of the parent expansion that produced
    node ``i`` (root carries ``(None, None)``), so a parent's ratio is read at
    its greedy child. The greedy top path has one parent per depth, so the
    first trigger along the descent is the earliest; the tie-break is defensive.
    """
    if raw_pairs is None or len(raw_pairs) != len(parents):
        return None
    best: Optional[TriggerInfo] = None
    for depth, child_index in enumerate(greedy_top_path(parents, logprobs)[1:]):
        pair = raw_pairs[child_index]
        if pair is None:
            continue
        z1, z2 = pair
        if z1 is None or z2 is None:
            continue
        ratio = float(z2) / (float(z1) + RATIO_EPS)
        if ratio <= theta:
            continue
        parent_index = parents[child_index]
        candidate = TriggerInfo(parent_index, depth, child_index, ratio)
        if best is None or (candidate.parent_depth, candidate.parent_index) < (
            best.parent_depth,
            best.parent_index,
        ):
            best = candidate
    return best
