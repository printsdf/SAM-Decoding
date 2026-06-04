from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..tree_model.fusion import TreeSpec


@dataclass(frozen=True)
class SamPrefixBudget:
    max_added_nodes: int
    top_k: int
    min_depth: int
    max_depth: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "SamPrefixBudget":
        return cls(
            max_added_nodes=config.sam_prefix_max_added_nodes,
            top_k=config.sam_prefix_top_k,
            min_depth=config.sam_prefix_min_depth,
            max_depth=config.sam_prefix_max_depth,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_added_nodes": self.max_added_nodes,
            "top_k": self.top_k,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
        }


@dataclass(frozen=True)
class PrefixAnchor:
    tree_index: int
    sam_index: int
    sam_length: int
    depth: int


def _node_depths(tree: TreeSpec) -> List[int]:
    depths = [0] * len(tree.tokens)
    for index in range(1, len(tree.tokens)):
        depths[index] = depths[tree.parents[index]] + 1
    return depths


def _transfer_direct(states: List[Any], index: int, length: int, token: int) -> Optional[Tuple[int, int]]:
    next_index = states[index].next.get(int(token))
    if next_index is None:
        return None
    return next_index, length + 1


def _cnt_endpos(state: Any) -> int:
    return int(getattr(state, "cnt_endpos", 0))


def _top_next_edges(states: List[Any], index: int, top_k: int) -> Tuple[List[Tuple[int, int]], int]:
    all_next = list(states[index].next.items())
    top_next = sorted(
        all_next,
        key=lambda item: (-_cnt_endpos(states[item[1]]), item[0]),
    )[:top_k]
    return top_next, max(0, len(all_next) - len(top_next))


def find_prefix_anchors(
    tree: TreeSpec,
    states: List[Any],
    start_index: int,
    start_length: int,
    budget: SamPrefixBudget,
) -> Tuple[List[PrefixAnchor], Dict[str, Any]]:
    depths = _node_depths(tree)
    leaf_indices = set(tree.leaf_indices())
    node_to_sam: Dict[int, Tuple[int, int]] = {0: (start_index, start_length)}
    anchors: List[PrefixAnchor] = []
    unmapped_node_count = 0
    skipped_leaf_anchor_count = 0
    skipped_depth_count = 0

    for tree_index in range(1, len(tree.tokens)):
        parent_index = tree.parents[tree_index]
        parent_sam = node_to_sam.get(parent_index)
        if parent_sam is None:
            unmapped_node_count += 1
            continue
        transferred = _transfer_direct(
            states,
            parent_sam[0],
            parent_sam[1],
            tree.tokens[tree_index],
        )
        if transferred is None:
            unmapped_node_count += 1
            continue
        node_to_sam[tree_index] = transferred

    for tree_index, (sam_index, sam_length) in node_to_sam.items():
        depth = depths[tree_index]
        if tree_index in leaf_indices:
            skipped_leaf_anchor_count += 1
            continue
        if depth < budget.min_depth:
            skipped_depth_count += 1
            continue
        if budget.max_depth is not None and depth > budget.max_depth:
            skipped_depth_count += 1
            continue
        anchors.append(
            PrefixAnchor(
                tree_index=tree_index,
                sam_index=sam_index,
                sam_length=sam_length,
                depth=depth,
            )
        )

    stats = {
        "mapped_node_count": len(node_to_sam),
        "unmapped_node_count": unmapped_node_count,
        "anchor_count": len(anchors),
        "skipped_leaf_anchor_count": skipped_leaf_anchor_count,
        "skipped_depth_count": skipped_depth_count,
    }
    return anchors, stats


def expand_local_siblings(
    base_tree: TreeSpec,
    anchors: List[PrefixAnchor],
    states: List[Any],
    budget: SamPrefixBudget,
) -> Tuple[TreeSpec, Dict[str, Any]]:
    tokens = list(base_tree.tokens)
    parents = list(base_tree.parents)
    leaf_indices = set(base_tree.leaf_indices())
    added_nodes = 0
    expanded_anchor_count = 0
    skipped_duplicate_sibling_count = 0
    skipped_leaf_anchor_count = 0
    skipped_budget_sibling_count = 0
    anchor_without_next_count = 0
    pruned_by_top_k = 0

    for anchor in anchors:
        if added_nodes >= budget.max_added_nodes:
            skipped_budget_sibling_count += len(states[anchor.sam_index].next)
            continue
        if anchor.tree_index in leaf_indices:
            skipped_leaf_anchor_count += 1
            continue

        next_edges, pruned = _top_next_edges(states, anchor.sam_index, budget.top_k)
        pruned_by_top_k += pruned
        if not next_edges:
            anchor_without_next_count += 1
            continue

        existing_child_tokens = {
            int(tokens[index])
            for index, parent in enumerate(parents)
            if parent == anchor.tree_index
        }
        added_for_anchor = 0
        for token, _ in next_edges:
            token = int(token)
            if token in existing_child_tokens:
                skipped_duplicate_sibling_count += 1
                continue
            if added_nodes >= budget.max_added_nodes:
                skipped_budget_sibling_count += 1
                continue
            tokens.append(token)
            parents.append(anchor.tree_index)
            existing_child_tokens.add(token)
            added_nodes += 1
            added_for_anchor += 1

        if added_for_anchor > 0:
            expanded_anchor_count += 1

    expanded_tree = TreeSpec(tokens=tokens, parents=parents)
    stats = {
        "expanded_anchor_count": expanded_anchor_count,
        "added_nodes": added_nodes,
        "node_count": len(tokens),
        "skipped_duplicate_sibling_count": skipped_duplicate_sibling_count,
        "skipped_leaf_anchor_count": skipped_leaf_anchor_count,
        "skipped_budget_sibling_count": skipped_budget_sibling_count,
        "anchor_without_next_count": anchor_without_next_count,
        "pruned_by_top_k": pruned_by_top_k,
    }
    return expanded_tree, stats
