"""Phase 2: Predicted-boundary SAM grafting.

Predict EAGLE's rejection depth from per-depth confidence margins,
then graft SAM continuation at that depth.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch

from samd.tree_model.fusion import TreeSpec
from .types import CandidateNode


def predict_rejection_depth(
    logprobs: Any,
    position_ids: torch.Tensor,
    threshold: float,
    min_depth: int = 3,
    max_depth: int = 8,
) -> int:
    """Predict first depth where EAGLE will likely be rejected.

    Computes per-depth average confidence (exp of logprob).
    Returns depth D where avg confidence at depth D < threshold.

    Args:
        logprobs: EAGLE draft logprobs (negative values), shape [num_nodes]
        position_ids: Tree position IDs to map nodes to depths, shape [num_nodes]
        threshold: Confidence threshold (0-1 range, lower = more uncertain)
        min_depth: Minimum depth to consider for prediction
        max_depth: Maximum depth to consider for prediction

    Returns:
        predicted_depth: Depth to graft SAM, or -1 if no prediction
    """
    if logprobs is None or len(logprobs) == 0:
        return -1

    # Convert to lists for simple iteration
    if hasattr(logprobs, 'tolist'):
        logprobs_list = logprobs.tolist() if hasattr(logprobs, 'tolist') else list(logprobs)
    elif isinstance(logprobs, torch.Tensor):
        logprobs_list = logprobs.cpu().tolist()
    else:
        logprobs_list = list(logprobs)

    if hasattr(position_ids, 'tolist'):
        position_list = position_ids.tolist() if hasattr(position_ids, 'tolist') else list(position_ids)
    elif isinstance(position_ids, torch.Tensor):
        position_list = position_ids.cpu().tolist()
    else:
        position_list = list(position_ids)

    # Handle nested lists
    if isinstance(logprobs_list, list) and len(logprobs_list) > 0 and isinstance(logprobs_list[0], list):
        logprobs_list = logprobs_list[0]
    if isinstance(position_list, list) and len(position_list) > 0 and isinstance(position_list[0], list):
        position_list = position_list[0]

    if len(logprobs_list) != len(position_list):
        return -1

    # Group confidence scores by depth (convert logprob to probability)
    import math
    depth_confidences: Dict[int, List[float]] = {}
    for logprob, depth in zip(logprobs_list, position_list):
        if min_depth <= depth <= max_depth:
            confidence = math.exp(float(logprob))  # Convert logprob to probability
            depth_confidences.setdefault(depth, []).append(confidence)

    # Find first depth where avg confidence < threshold
    for depth in sorted(depth_confidences.keys()):
        confidences = depth_confidences[depth]
        avg_confidence = sum(confidences) / len(confidences)
        if avg_confidence < threshold:
            return depth

    return -1


def extract_prefix_tokens_and_node(
    eagle_tree: TreeSpec,
    graft_depth: int,
) -> Tuple[List[int], int]:
    """Extract EAGLE prefix tokens and node index up to graft_depth-1.

    Returns tokens at depth 0 to graft_depth-1 and the node index at graft_depth-1
    (for SAM transfer_state and grafting).

    Args:
        eagle_tree: EAGLE tree structure
        graft_depth: Depth at which SAM will be grafted

    Returns:
        (prefix_tokens, parent_node_idx): Prefix sequence and node to graft onto
    """
    if graft_depth <= 0:
        return [], 0

    # Compute depths from parents
    depths = [0] * len(eagle_tree.tokens)
    for i in range(1, len(eagle_tree.tokens)):
        parent = eagle_tree.parents[i]
        depths[i] = depths[parent] + 1

    # Find first sequence path up to graft_depth-1
    prefix = [eagle_tree.tokens[0]]  # Root token
    current_idx = 0

    for target_depth in range(1, graft_depth):
        # Find first child of current_idx at target_depth
        found = False
        for i in range(current_idx + 1, len(eagle_tree.tokens)):
            if eagle_tree.parents[i] == current_idx and depths[i] == target_depth:
                prefix.append(eagle_tree.tokens[i])
                current_idx = i
                found = True
                break
        if not found:
            # Could not reach target depth, return what we have
            break

    return prefix, current_idx


def find_leaf_nodes_at_depth(
    eagle_tree: TreeSpec,
) -> Dict[int, List[int]]:
    """Find leaf node indices grouped by depth.

    Args:
        eagle_tree: EAGLE tree structure

    Returns:
        leaf_map: Dict mapping depth -> list of leaf node indices
    """
    # Compute depths
    depths = [0] * len(eagle_tree.tokens)
    for i in range(1, len(eagle_tree.tokens)):
        parent = eagle_tree.parents[i]
        depths[i] = depths[parent] + 1

    # Find leaf nodes (nodes with no children)
    is_leaf = [True] * len(eagle_tree.tokens)
    for i in range(1, len(eagle_tree.tokens)):
        parent = eagle_tree.parents[i]
        is_leaf[parent] = False

    # Group by depth
    leaf_map: Dict[int, List[int]] = {}
    for i, (leaf, depth) in enumerate(zip(is_leaf, depths)):
        if leaf:
            leaf_map.setdefault(depth, []).append(i)

    return leaf_map


def graft_sam_at_depth(
    eagle_tree: TreeSpec,
    sam_candidates: List[int],
    parent_node_idx: int,
    max_added_nodes: int,
) -> TreeSpec:
    """Graft SAM continuation onto specific EAGLE node.

    Args:
        eagle_tree: EAGLE tree structure
        sam_candidates: SAM generated token sequence
        parent_node_idx: EAGLE node index to graft onto (from extract_prefix_tokens_and_node)
        max_added_nodes: Max SAM nodes to add

    Returns:
        fused_tree: TreeSpec with SAM tail grafted
    """
    if not sam_candidates or parent_node_idx < 0:
        return eagle_tree

    # Start with EAGLE tokens and parents
    new_tokens = list(eagle_tree.tokens)
    new_parents = list(eagle_tree.parents)

    # Graft SAM tokens onto the specified parent node
    current_parent = parent_node_idx

    sam_to_add = sam_candidates[:max_added_nodes]
    for sam_token in sam_to_add:
        new_tokens.append(sam_token)
        new_parents.append(current_parent)
        current_parent = len(new_tokens) - 1  # Chain SAM tokens sequentially

    return TreeSpec(tokens=new_tokens, parents=new_parents)


def graft_sam_nodes_as_candidates(
    eagle_nodes: List[CandidateNode],
    sam_candidates: List[int],
    graft_depth: int,
    max_added_nodes: int,
) -> List[CandidateNode]:
    """Add SAM continuation nodes to EAGLE candidate list.

    Args:
        eagle_nodes: Existing EAGLE candidate nodes
        sam_candidates: SAM token sequence
        graft_depth: Depth at which to start SAM
        max_added_nodes: Max SAM nodes to add

    Returns:
        fused_nodes: Combined candidate list
    """
    # Find path at graft_depth-1 for SAM prefix
    prefix_path = []
    for node in eagle_nodes:
        if node.depth == graft_depth - 1:
            prefix_path = node.path + [node.token]
            break

    if not prefix_path and graft_depth > 0:
        # No valid prefix found, return eagle only
        return eagle_nodes

    # Create SAM candidate nodes
    sam_nodes = []
    current_path = prefix_path if graft_depth > 0 else []

    sam_to_add = sam_candidates[:max_added_nodes]
    for i, sam_token in enumerate(sam_to_add):
        sam_nodes.append(
            CandidateNode(
                token=sam_token,
                source="sam",
                score=1.0 / (graft_depth + i + 1),  # Depth-based score
                depth=graft_depth + i,
                path=current_path,
                source_scores={"sam": 1.0},
            )
        )
        current_path = current_path + [sam_token]

    return eagle_nodes + sam_nodes
