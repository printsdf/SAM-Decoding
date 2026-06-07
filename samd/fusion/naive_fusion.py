from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from samd.tree_model.fusion import TreeSpec

from .types import CandidateNode, FusedTree, FusionConfig
from .utils import source_counts


def _squeeze_tree_tokens(tokens: Any) -> List[int]:
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu()
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    while isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


def _squeeze_logprobs(logprobs: Any) -> List[float]:
    if logprobs is None:
        return []
    if hasattr(logprobs, "detach"):
        logprobs = logprobs.detach().cpu()
    if hasattr(logprobs, "tolist"):
        logprobs = logprobs.tolist()
    while (
        isinstance(logprobs, list)
        and len(logprobs) == 1
        and isinstance(logprobs[0], list)
    ):
        logprobs = logprobs[0]
    return [float(value) for value in logprobs]


def _normalization_groups(nodes: Iterable[CandidateNode]) -> Dict[str, List[float]]:
    groups: Dict[str, List[float]] = {}
    for node in nodes:
        groups.setdefault(node.source, []).append(float(node.score))
    return groups


def _normalize_score(score: float, values: List[float]) -> float:
    if not values:
        return 0.0
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return 1.0
    return (float(score) - min_value) / (max_value - min_value)


def _normalize_by_source(nodes: List[CandidateNode]) -> List[CandidateNode]:
    groups = _normalization_groups(nodes)
    for node in nodes:
        node.normalized_score = _normalize_score(node.score, groups[node.source])
    return nodes


def _safe_mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _count_duplicates(
    eagle_nodes: List[CandidateNode],
    sam_nodes: List[CandidateNode],
) -> int:
    eagle_keys = {node.key for node in eagle_nodes}
    sam_keys = {node.key for node in sam_nodes}
    return len(eagle_keys & sam_keys)


def _node_sources_by_tree_index(fused_tree: FusedTree) -> List[str]:
    """Return source labels aligned with fused_tree token indices."""
    return ["root"] + [node.source for node in fused_tree.nodes]


def parse_eagle_tree(
    eagle_tree: Dict[str, Any],
    start_token: int,
    eagle_logprobs: Optional[Any] = None,
) -> List[CandidateNode]:
    """Convert EAGLE3 flattened tree buffers into non-root candidate nodes."""
    tokens = _squeeze_tree_tokens(eagle_tree["tokens"])
    if not tokens:
        return []
    if int(tokens[0]) != int(start_token):
        raise ValueError("EAGLE tree root token does not match start_token")

    tree_spec = TreeSpec.from_eagle3_buffers(
        tokens,
        eagle_tree["tree_attn_mask"],
        eagle_tree["tree_position_ids"],
    )
    raw_logprobs = eagle_logprobs if eagle_logprobs is not None else eagle_tree.get("logprobs")
    logprobs = _squeeze_logprobs(raw_logprobs)
    if raw_logprobs is not None and len(logprobs) != len(tree_spec.tokens):
        raise ValueError("EAGLE logprobs length must match draft_tokens length")

    nodes: List[CandidateNode] = []
    for index in range(1, len(tree_spec.tokens)):
        parent_index = tree_spec.parents[index]
        path_indices = tree_spec.node_path_indices(parent_index)
        path = [int(tree_spec.tokens[path_index]) for path_index in path_indices]
        depth = len(path)
        score = (
            float(logprobs[index])
            if raw_logprobs is not None
            else 1.0 / float(depth + 1)
        )
        nodes.append(
            CandidateNode(
                token=int(tree_spec.tokens[index]),
                source="eagle",
                score=score,
                depth=depth,
                path=path,
                source_scores={"eagle": score},
            )
        )
    return nodes


def parse_sam_sequence(
    sam_candidates: List[int],
    start_token: int,
    match_length: Optional[int] = None,
) -> List[CandidateNode]:
    """Convert a linear SAM draft sequence into non-root candidate nodes."""
    if not sam_candidates:
        return []
    if int(sam_candidates[0]) != int(start_token):
        raise ValueError("SAM candidates must start with start_token")

    score_horizon = (
        int(match_length)
        if match_length is not None and int(match_length) > 0
        else len(sam_candidates)
    )
    nodes: List[CandidateNode] = []
    path = [int(start_token)]
    for position, token in enumerate(sam_candidates[1:], start=1):
        score = max(float(score_horizon - position), 0.0)
        nodes.append(
            CandidateNode(
                token=int(token),
                source="sam",
                score=score,
                depth=position,
                path=list(path),
                source_scores={"sam": score},
            )
        )
        path.append(int(token))
    return nodes


def merge_and_dedup(
    eagle_nodes: List[CandidateNode],
    sam_nodes: List[CandidateNode],
    strategy: str,
) -> List[CandidateNode]:
    """Merge EAGLE and SAM nodes, optionally deduplicating identical tree positions."""
    all_nodes = _normalize_by_source([deepcopy(node) for node in eagle_nodes + sam_nodes])
    if strategy == "keep_both":
        return all_nodes
    if strategy not in ("max_score", "sum_score"):
        raise ValueError("unsupported fusion dedup strategy: {}".format(strategy))

    by_key: Dict[Tuple[Tuple[int, ...], int], CandidateNode] = {}
    for node in all_nodes:
        key = node.key
        if key not in by_key:
            by_key[key] = node
            continue

        current = by_key[key]
        if strategy == "max_score":
            if node.normalized_score > current.normalized_score:
                replacement = node
                replacement.source_scores = {
                    **current.source_scores,
                    **replacement.source_scores,
                }
                replacement.metadata = {**current.metadata, **replacement.metadata}
                by_key[key] = replacement
            else:
                current.source_scores.update(node.source_scores)
            if current.source != node.source:
                by_key[key].source = "both"
                by_key[key].normalized_score = max(
                    current.normalized_score,
                    node.normalized_score,
                )
        else:
            current.score += node.score
            current.normalized_score += node.normalized_score
            current.source_scores.update(node.source_scores)
            if current.source != node.source:
                current.source = "both"

    return list(by_key.values())


def sort_by_score(nodes: List[CandidateNode]) -> List[CandidateNode]:
    """Sort nodes by normalized score, then by shallow depth and stable token path."""
    return sorted(
        nodes,
        key=lambda node: (
            -float(node.normalized_score),
            int(node.depth),
            tuple(node.path),
            int(node.token),
        ),
    )


def sort_depth_first(nodes: List[CandidateNode]) -> List[CandidateNode]:
    """Sort nodes by prefix order, using score only to order siblings."""
    return sorted(
        nodes,
        key=lambda node: (
            int(node.depth),
            tuple(node.path),
            -float(node.normalized_score),
            int(node.token),
        ),
    )


def _ancestor_keys(node: CandidateNode) -> List[Tuple[Tuple[int, ...], int]]:
    ancestors = []
    for index in range(1, len(node.path)):
        parent_path = tuple(node.path[:index])
        token = int(node.path[index])
        ancestors.append((parent_path, token))
    return ancestors


def truncate_with_ancestors(
    sorted_nodes: List[CandidateNode],
    max_tokens: int,
) -> List[CandidateNode]:
    """Select up to max_tokens non-root nodes while preserving prefix closure."""
    if max_tokens <= 0:
        return []

    by_key: Dict[Tuple[Tuple[int, ...], int], CandidateNode] = {}
    for node in sorted_nodes:
        by_key.setdefault(node.key, node)
    selected: List[CandidateNode] = []
    selected_keys = set()

    for node in sorted_nodes:
        needed_keys = [key for key in _ancestor_keys(node) if key not in selected_keys]
        needs_self = node.key not in selected_keys or node.source != "both"
        if needs_self:
            needed_keys.append(node.key)
        if len(selected) + len(needed_keys) > max_tokens:
            continue
        missing = [key for key in needed_keys if key not in by_key]
        if missing:
            continue
        for key in needed_keys:
            selected_node = node if key == node.key else by_key[key]
            selected.append(selected_node)
            selected_keys.add(key)

    return sort_by_score(selected)


def _order_prefix_closed_nodes(nodes: List[CandidateNode]) -> List[CandidateNode]:
    remaining = list(nodes)
    ordered: List[CandidateNode] = []
    emitted_paths = {tuple([node.path[0]]) for node in nodes if node.path}

    while remaining:
        progressed = False
        for node in list(remaining):
            parent_path = tuple(node.path)
            if len(parent_path) == 1 or parent_path in emitted_paths:
                ordered.append(node)
                emitted_paths.add(node.token_path)
                remaining.remove(node)
                progressed = True
        if not progressed:
            missing = [list(node.path) + [node.token] for node in remaining]
            raise ValueError("selected nodes are not prefix-closed: {}".format(missing))
    return ordered


def build_tree_buffers(
    nodes: List[CandidateNode],
    start_token: int,
    device: Optional[torch.device] = None,
    mask_dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FusedTree:
    """Build verifier-compatible tree buffers from selected non-root nodes."""
    ordered_nodes = _order_prefix_closed_nodes(nodes)
    tokens = [int(start_token)]
    parents = [-1]
    path_to_index = {(int(start_token),): 0}
    exported_nodes: List[CandidateNode] = []

    for node in ordered_nodes:
        if not node.path or int(node.path[0]) != int(start_token):
            raise ValueError("candidate node path must start with start_token")
        parent_path = tuple(int(token) for token in node.path)
        if parent_path not in path_to_index:
            raise ValueError("missing parent path while building fusion tree")
        parent_index = path_to_index[parent_path]
        node = deepcopy(node)
        node.parent_idx = parent_index
        tokens.append(int(node.token))
        parents.append(parent_index)
        path_to_index.setdefault(node.token_path, len(tokens) - 1)
        exported_nodes.append(node)

    tree_spec = TreeSpec(tokens=tokens, parents=parents)
    buffers_kwargs = tree_spec.to_buffers(device=device, mask_dtype=mask_dtype)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
    tree_mask = buffers_kwargs["tree_attn_mask"]
    position_ids = buffers_kwargs["tree_position_ids"]
    retrieve_indices = buffers_kwargs["tree_retrieve_indices"]
    return FusedTree(
        nodes=exported_nodes,
        tokens=tokens_tensor,
        tree_mask=tree_mask,
        position_ids=position_ids,
        retrieve_indices=retrieve_indices,
        buffers_kwargs=buffers_kwargs,
        metadata=metadata or {},
    )


def fuse_eagle_sam_naive(
    eagle_tree: Dict[str, Any],
    sam_candidates: List[int],
    start_token: int,
    config: FusionConfig,
    sam_match_length: Optional[int] = None,
    eagle_logprobs: Optional[Any] = None,
) -> FusedTree:
    """Naive EAGLE3 + SAM fusion: merge, deduplicate, score-sort, and truncate."""
    eagle_nodes = parse_eagle_tree(eagle_tree, start_token, eagle_logprobs=eagle_logprobs)
    sam_nodes = parse_sam_sequence(sam_candidates, start_token, sam_match_length)
    merged_nodes = merge_and_dedup(
        eagle_nodes,
        sam_nodes,
        strategy=config.dedup_strategy,
    )
    if config.truncate_strategy == "score":
        sorted_nodes = sort_by_score(merged_nodes)
    elif config.truncate_strategy == "depth_first":
        sorted_nodes = sort_depth_first(merged_nodes)
    else:
        raise ValueError("unsupported fusion truncate strategy: {}".format(config.truncate_strategy))

    max_tokens = int(config.max_draft_tokens)
    if max_tokens <= 0:
        selected_nodes = []
    elif len(sorted_nodes) > max_tokens:
        selected_nodes = truncate_with_ancestors(sorted_nodes, max_tokens)
    else:
        selected_nodes = sorted_nodes

    tree_attn_mask = eagle_tree.get("tree_attn_mask")
    tree_position_ids = eagle_tree.get("tree_position_ids")
    device = None
    mask_dtype = None
    if hasattr(tree_position_ids, "device"):
        device = tree_position_ids.device
    if hasattr(tree_attn_mask, "dtype"):
        mask_dtype = tree_attn_mask.dtype

    sam_match_length_value = max(int(sam_match_length or 0), 0)
    final_source_counts = source_counts(selected_nodes)
    dedup_count = len(eagle_nodes) + len(sam_nodes) - len(merged_nodes)
    metadata = {
        "mode": config.mode,
        "eagle_nodes": len(eagle_nodes),
        "sam_nodes": len(sam_nodes),
        "merged_nodes": len(merged_nodes),
        "final_nodes": len(selected_nodes),
        "eagle_avg_score": _safe_mean(node.score for node in eagle_nodes),
        "sam_avg_score": _safe_mean(node.score for node in sam_nodes),
        "eagle_avg_depth": _safe_mean(node.depth for node in eagle_nodes),
        "sam_avg_depth": _safe_mean(node.depth for node in sam_nodes),
        "sam_avg_match_length": float(sam_match_length_value) if sam_nodes else 0.0,
        "sam_max_match_length": sam_match_length_value if sam_nodes else 0,
        "dedup_count": dedup_count,
        "both_proposed_count": _count_duplicates(eagle_nodes, sam_nodes),
        "truncate_count": max(len(merged_nodes) - len(selected_nodes), 0),
        "truncated_count": max(len(merged_nodes) - len(selected_nodes), 0),
        "selected_eagle": sum(1 for node in selected_nodes if node.source == "eagle"),
        "selected_sam": sum(1 for node in selected_nodes if node.source == "sam"),
        "selected_both": sum(1 for node in selected_nodes if node.source == "both"),
        "eagle_contribution": final_source_counts["eagle"],
        "sam_contribution": final_source_counts["sam"],
        "both_contribution": final_source_counts["both"],
        "max_draft_tokens": max_tokens,
        "dedup_strategy": config.dedup_strategy,
        "truncate_strategy": config.truncate_strategy,
        "sam_match_length": sam_match_length_value,
    }
    metadata.update(config.metadata)

    fused_tree = build_tree_buffers(
        selected_nodes,
        start_token,
        device=device,
        mask_dtype=mask_dtype,
        metadata={},
    )
    metadata["node_sources"] = _node_sources_by_tree_index(fused_tree)
    fused_tree.metadata.update(metadata)
    return fused_tree
