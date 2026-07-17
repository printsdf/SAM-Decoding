"""Fixed-budget pruning for drafter-MARS fused trees (pure, torch-free)."""

import heapq
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..tree_model.fusion import TreeSpec


def _ancestor_closure(parents: Sequence[int], indices: Iterable[int]) -> Set[int]:
    protected: Set[int] = {0}
    for index in indices:
        node = int(index)
        while node != -1 and node not in protected:
            protected.add(node)
            node = int(parents[node])
    return protected


def prune_eagle_leaves_to_budget(
    tree: TreeSpec,
    *,
    eagle_node_count: int,
    eagle_logprobs: Sequence[float],
    max_total_nodes: int,
    protected_indices: Optional[Iterable[int]] = None,
) -> Tuple[TreeSpec, Dict[str, Any]]:
    """Prune low-value EAGLE leaves while preserving grafted SAM paths.

    ``eagle_node_count`` is the root-inclusive size of the original EAGLE tree.
    Nodes appended by grafting are never direct pruning candidates; their
    ancestor closure is protected so every SAM path remains valid. EAGLE leaves
    are removed one at a time by ascending mean path logprob until the fused
    tree fits ``max_total_nodes``.
    """
    if not isinstance(max_total_nodes, int) or isinstance(max_total_nodes, bool):
        raise ValueError("max_total_nodes must be a positive integer")
    if max_total_nodes < 1:
        raise ValueError("max_total_nodes must be a positive integer")
    if eagle_node_count < 1 or eagle_node_count > len(tree.tokens):
        raise ValueError("eagle_node_count must cover the original EAGLE tree")
    if len(eagle_logprobs) != eagle_node_count:
        raise ValueError("eagle_logprobs must align with the original EAGLE tree")

    original_total = len(tree.tokens)
    if original_total <= max_total_nodes:
        return tree, {
            "tree_budget": int(max_total_nodes),
            "pre_prune_nodes": original_total,
            "pruned_eagle_nodes": 0,
            "final_total_nodes": original_total,
            "removed_eagle_indices": [],
        }

    protected_seed = set(int(index) for index in (protected_indices or ()))
    protected_seed.update(range(eagle_node_count, original_total))
    protected = _ancestor_closure(tree.parents, protected_seed)

    children: List[Set[int]] = [set() for _ in tree.tokens]
    depths = [0] * original_total
    for index in range(1, original_total):
        parent = tree.parents[index]
        children[parent].add(index)
        depths[index] = depths[parent] + 1

    cumulative = [0.0] * eagle_node_count
    mean_path_logprob = [float("inf")] * eagle_node_count
    for index in range(1, eagle_node_count):
        cumulative[index] = cumulative[tree.parents[index]] + float(
            eagle_logprobs[index]
        )
        mean_path_logprob[index] = cumulative[index] / float(depths[index])

    active = [True] * original_total
    heap: List[Tuple[float, int, int, int]] = []

    def offer(index: int) -> None:
        if (
            index <= 0
            or index >= eagle_node_count
            or index in protected
            or not active[index]
            or children[index]
        ):
            return
        heapq.heappush(
            heap,
            (mean_path_logprob[index], -depths[index], -index, index),
        )

    for index in range(1, eagle_node_count):
        offer(index)

    active_count = original_total
    removed: List[int] = []
    while active_count > max_total_nodes:
        while heap:
            _score, _neg_depth, _neg_index, index = heapq.heappop(heap)
            if active[index] and not children[index] and index not in protected:
                break
        else:
            raise ValueError(
                "cannot satisfy drafter-MARS tree budget without pruning a protected or SAM node"
            )

        active[index] = False
        active_count -= 1
        removed.append(index)
        parent = tree.parents[index]
        children[parent].remove(index)
        offer(parent)

    old_to_new: Dict[int, int] = {}
    tokens: List[int] = []
    parents: List[int] = []
    for old_index, is_active in enumerate(active):
        if not is_active:
            continue
        new_index = len(tokens)
        old_to_new[old_index] = new_index
        tokens.append(tree.tokens[old_index])
        parents.append(-1 if old_index == 0 else old_to_new[tree.parents[old_index]])

    pruned = TreeSpec(tokens=tokens, parents=parents)
    return pruned, {
        "tree_budget": int(max_total_nodes),
        "pre_prune_nodes": original_total,
        "pruned_eagle_nodes": len(removed),
        "final_total_nodes": len(pruned.tokens),
        "removed_eagle_indices": removed,
    }
