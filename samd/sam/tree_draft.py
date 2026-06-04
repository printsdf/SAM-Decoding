import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..tree_model.fusion import TreeSpec


@dataclass(frozen=True)
class SamTreeBudget:
    max_nodes: int
    top_k: int
    alpha: float
    max_depth: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "SamTreeBudget":
        return cls(
            max_nodes=config.sam_tree_max_nodes,
            top_k=config.sam_tree_top_k,
            alpha=config.sam_tree_alpha,
            max_depth=config.sam_tree_max_depth,
        )

    def target_nodes(self, match_length: int) -> int:
        return min(self.max_nodes, max(1, 1 + int(match_length * self.alpha)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_nodes": self.max_nodes,
            "top_k": self.top_k,
            "alpha": self.alpha,
            "max_depth": self.max_depth,
        }


@dataclass(order=True)
class SearchItem:
    prob: float
    token: int = field(compare=False)
    index: int = field(compare=False)
    anc_tree_index: int = field(compare=False)
    depth: int = field(compare=False)


def _cnt_endpos(state: Any) -> int:
    return int(getattr(state, "cnt_endpos", 0))


def _top_next_edges(states: List[Any], index: int, top_k: int) -> Tuple[List[Tuple[int, int]], int]:
    all_next = list(states[index].next.items())
    top_next = sorted(
        all_next,
        key=lambda item: (-_cnt_endpos(states[item[1]]), item[0]),
    )[:top_k]
    return top_next, max(0, len(all_next) - len(top_next))


def build_sam_tree(
    states: List[Any],
    index: int,
    match_length: int,
    start_token: int,
    budget: SamTreeBudget,
) -> Tuple[TreeSpec, Dict[str, Any]]:
    target_nodes = budget.target_nodes(match_length)
    heap: List[SearchItem] = []
    tokens: List[int] = []
    parents: List[int] = []
    per_depth_count: Dict[int, int] = {}
    pruned_by_top_k = 0
    depth_limit_hit = False

    heapq.heappush(
        heap,
        SearchItem(
            prob=-1.0,
            token=int(start_token),
            index=index,
            anc_tree_index=-1,
            depth=0,
        ),
    )

    while len(tokens) < target_nodes and heap:
        item = heapq.heappop(heap)
        if per_depth_count.get(item.depth, 0) >= budget.top_k:
            continue
        per_depth_count[item.depth] = per_depth_count.get(item.depth, 0) + 1

        cur_tree_index = len(tokens)
        tokens.append(int(item.token))
        parents.append(item.anc_tree_index)

        if len(tokens) >= target_nodes:
            break
        if budget.max_depth is not None and item.depth >= budget.max_depth:
            if states[item.index].next:
                depth_limit_hit = True
            continue

        next_edges, pruned = _top_next_edges(states, item.index, budget.top_k)
        pruned_by_top_k += pruned
        if not next_edges:
            continue

        cnt_sum = sum(max(_cnt_endpos(states[next_index]), 1) for _, next_index in next_edges)
        for next_token, next_index in next_edges:
            next_prob = max(_cnt_endpos(states[next_index]), 1) / cnt_sum
            heapq.heappush(
                heap,
                SearchItem(
                    prob=item.prob * next_prob,
                    token=int(next_token),
                    index=next_index,
                    anc_tree_index=cur_tree_index,
                    depth=item.depth + 1,
                ),
            )

    stats = {
        "target_nodes": target_nodes,
        "node_count": len(tokens),
        "budget_hit": bool(
            (len(tokens) >= target_nodes and len(heap) > 0)
            or pruned_by_top_k > 0
            or depth_limit_hit
        ),
        "pruned_by_top_k": pruned_by_top_k,
        "depth_limit_hit": depth_limit_hit,
    }
    return TreeSpec(tokens=tokens, parents=parents), stats
