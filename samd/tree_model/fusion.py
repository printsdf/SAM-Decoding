from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _pad_path(path: List[int], length: int, pad_value: int = -1) -> List[int]:
    return path + [pad_value] * (length - len(path))


def _as_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _squeeze_singletons(value: Any) -> Any:
    value = _as_list(value)
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value


def eagle3_parents_from_buffers(
    tree_attn_mask: Any,
    tree_position_ids: Any,
) -> List[int]:
    """Torch equivalent of ``TreeSpec.from_eagle3_buffers`` parent derivation.

    Avoids the O(n^2) Python mask parse on the per-step hot path; the only
    device-to-host transfer is the final parent list. Selection rule matches
    ``from_eagle3_buffers`` exactly: parent of node i is the LARGEST j < i
    with mask[i, j] set and position[j] == position[i] - 1.
    """
    import torch

    pos = torch.as_tensor(tree_position_ids).reshape(-1).long()
    n = int(pos.numel())
    if n == 0:
        raise ValueError("TreeSpec must contain at least the root token")
    mask = torch.as_tensor(tree_attn_mask).reshape(n, n).bool()
    if bool((pos[1:] <= 0).any()):
        raise ValueError("non-root tree node must have positive position id")
    idx = torch.arange(n, device=pos.device)
    candidates = (
        mask
        & (pos.unsqueeze(0) == pos.unsqueeze(1) - 1)
        & (idx.unsqueeze(0) < idx.unsqueeze(1))
    )
    parents = (candidates.long() * (idx + 1).unsqueeze(0)).amax(dim=1) - 1
    parents[0] = -1
    if bool((parents[1:] < 0).any()):
        raise ValueError("could not infer parent for tree node")
    return parents.tolist()


@dataclass
class TreeSpec:
    tokens: List[int]
    parents: List[int]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.parents):
            raise ValueError("TreeSpec tokens and parents must have the same length")
        if not self.tokens:
            raise ValueError("TreeSpec must contain at least the root token")
        if self.parents[0] != -1:
            raise ValueError("TreeSpec root parent must be -1")
        for index, parent in enumerate(self.parents[1:], start=1):
            if parent < 0 or parent >= index:
                raise ValueError(
                    "TreeSpec parent at index {} must point to an earlier node".format(index)
                )

    @classmethod
    def from_eagle3_buffers(
        cls,
        draft_tokens: Any,
        tree_attn_mask: Any,
        tree_position_ids: Any,
    ) -> "TreeSpec":
        tokens = [int(token) for token in _squeeze_singletons(draft_tokens)]
        positions = [int(position) for position in _squeeze_singletons(tree_position_ids)]
        mask = _squeeze_singletons(tree_attn_mask)

        n_nodes = len(tokens)
        if len(positions) != n_nodes:
            raise ValueError("tree_position_ids length must match draft_tokens length")
        if len(mask) != n_nodes or any(len(row) != n_nodes for row in mask):
            raise ValueError("tree_attn_mask must be an NxN matrix after squeezing")

        parents = [-1]
        for index in range(1, n_nodes):
            depth = positions[index]
            if depth <= 0:
                raise ValueError("non-root tree node must have positive position id")
            candidates = [
                candidate
                for candidate in range(index)
                if bool(mask[index][candidate]) and positions[candidate] == depth - 1
            ]
            if not candidates:
                raise ValueError("could not infer parent for tree node {}".format(index))
            parents.append(candidates[-1])
        return cls(tokens=tokens, parents=parents)

    def to_buffer_lists(self) -> Tuple[List[List[bool]], List[int], List[List[int]]]:
        n_nodes = len(self.tokens)
        is_leaf = [True] * n_nodes
        position_ids = [0] * n_nodes

        for index in range(1, n_nodes):
            parent = self.parents[index]
            is_leaf[parent] = False
            position_ids[index] = position_ids[parent] + 1

        tree_attn_mask = [[False] * n_nodes for _ in range(n_nodes)]
        for index in range(n_nodes):
            cur = index
            while cur != -1:
                tree_attn_mask[index][cur] = True
                cur = self.parents[cur]

        retrieve_indices = []
        for index in range(n_nodes):
            if not is_leaf[index]:
                continue
            path = [index]
            while path[-1] != 0:
                path.append(self.parents[path[-1]])
            retrieve_indices.append(list(reversed(path)))

        max_depth = max(len(path) for path in retrieve_indices)
        retrieve_indices = [_pad_path(path, max_depth) for path in retrieve_indices]
        return tree_attn_mask, position_ids, retrieve_indices

    def child_indices(self, parent_index: int) -> List[int]:
        if parent_index < 0 or parent_index >= len(self.tokens):
            raise IndexError("parent_index out of range")
        return [index for index, parent in enumerate(self.parents) if parent == parent_index]

    def leaf_indices(self) -> List[int]:
        has_child = [False] * len(self.tokens)
        for parent in self.parents[1:]:
            has_child[parent] = True
        return [index for index, child_exists in enumerate(has_child) if not child_exists]

    def node_path_indices(self, index: int) -> List[int]:
        if index < 0 or index >= len(self.tokens):
            raise IndexError("index out of range")
        path = [index]
        while path[-1] != 0:
            path.append(self.parents[path[-1]])
        return list(reversed(path))

    def node_token_path(self, index: int) -> Tuple[int, ...]:
        return tuple(self.tokens[path_index] for path_index in self.node_path_indices(index))

    def leaf_token_paths(self) -> List[Tuple[int, ...]]:
        return [self.node_token_path(index) for index in self.leaf_indices()]

    def missing_leaf_token_paths(self, baseline: "TreeSpec") -> List[Tuple[int, ...]]:
        expanded_leaf_paths = set(self.leaf_token_paths())
        return [
            path
            for path in baseline.leaf_token_paths()
            if path not in expanded_leaf_paths
        ]

    def leaf_retention_stats(self, baseline: "TreeSpec") -> Dict[str, Any]:
        baseline_leaf_paths = baseline.leaf_token_paths()
        missing_leaf_paths = self.missing_leaf_token_paths(baseline)
        return {
            "baseline_leaf_count": len(baseline_leaf_paths),
            "retained_leaf_count": len(baseline_leaf_paths) - len(missing_leaf_paths),
            "missing_leaf_count": len(missing_leaf_paths),
            "missing_leaf_paths": [list(path) for path in missing_leaf_paths],
        }

    def assert_leaf_paths_retained(
        self,
        baseline: "TreeSpec",
        context: str = "TreeSpec leaf retention",
    ) -> None:
        missing_leaf_paths = self.missing_leaf_token_paths(baseline)
        if missing_leaf_paths:
            formatted_paths = [list(path) for path in missing_leaf_paths]
            raise ValueError(
                "{} failed; missing leaf paths: {}".format(context, formatted_paths)
            )

    def graft_sequence(self, sequence: List[int]) -> "TreeSpec":
        if not sequence:
            return TreeSpec(tokens=list(self.tokens), parents=list(self.parents))
        if int(sequence[0]) != self.tokens[0]:
            raise ValueError("grafted sequence must start with the TreeSpec root token")

        tokens = list(self.tokens)
        parents = list(self.parents)
        parent = 0
        for token in sequence[1:]:
            token = int(token)
            child = None
            for index in range(parent + 1, len(tokens)):
                if parents[index] == parent and tokens[index] == token:
                    child = index
                    break
            if child is None:
                child = len(tokens)
                tokens.append(token)
                parents.append(parent)
            parent = child
        return TreeSpec(tokens=tokens, parents=parents)

    def union_sam_tree(self, sam_tree: "TreeSpec") -> Tuple["TreeSpec", Dict[str, int]]:
        if int(sam_tree.tokens[0]) != self.tokens[0]:
            raise ValueError("union SAM tree must start with the TreeSpec root token")

        tokens = list(self.tokens)
        parents = list(self.parents)
        sam_to_fused = {0: 0}
        sam_added_nodes = 0
        merged_nodes = 0

        for sam_index in range(1, len(sam_tree.tokens)):
            sam_parent = sam_tree.parents[sam_index]
            fused_parent = sam_to_fused[sam_parent]
            token = int(sam_tree.tokens[sam_index])
            child = None
            for index in range(fused_parent + 1, len(tokens)):
                if parents[index] == fused_parent and tokens[index] == token:
                    child = index
                    break
            if child is None:
                child = len(tokens)
                tokens.append(token)
                parents.append(fused_parent)
                sam_added_nodes += 1
            else:
                merged_nodes += 1
            sam_to_fused[sam_index] = child

        stats = {
            "eagle_node_count": len(self.tokens),
            "sam_node_count": len(sam_tree.tokens),
            "node_count": len(tokens),
            "sam_added_nodes": sam_added_nodes,
            "merged_nodes": merged_nodes,
        }
        return TreeSpec(tokens=tokens, parents=parents), stats

    def to_buffers(
        self,
        device: Optional[Any] = None,
        mask_dtype: Optional[Any] = None,
    ) -> Dict[str, Any]:
        import torch

        tree_attn_mask, position_ids, retrieve_indices = self.to_buffer_lists()
        attn_mask_tensor = torch.tensor(
            tree_attn_mask,
            dtype=torch.bool,
            device=device,
        ).view(1, 1, len(self.tokens), len(self.tokens))
        if mask_dtype is not None and mask_dtype != torch.bool:
            attn_mask_tensor = attn_mask_tensor.to(mask_dtype)
        return {
            "tree_attn_mask": attn_mask_tensor,
            "tree_position_ids": torch.tensor(
                [position_ids],
                dtype=torch.long,
                device=device,
            ),
            "tree_retrieve_indices": torch.tensor(
                retrieve_indices,
                dtype=torch.long,
                device=device,
            ),
        }

    def to_draft(
        self,
        device: Optional[Any] = None,
        mask_dtype: Optional[Any] = None,
    ) -> Tuple[List[int], Dict[str, Any]]:
        return list(self.tokens), self.to_buffers(device=device, mask_dtype=mask_dtype)
