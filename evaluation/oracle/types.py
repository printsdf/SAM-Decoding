"""Shared same-trace oracle types."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def int_tuple(value: Any) -> Tuple[int, ...]:
    if value is None:
        return ()
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    flattened: List[int] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            flattened.extend(int_tuple(item))
        else:
            flattened.append(int(item))
    return tuple(flattened)


@dataclass(frozen=True)
class OracleCandidate:
    source: str
    token: int
    depth: int
    score: float
    accepted: bool
    path: Tuple[int, ...]
    token_path: Tuple[int, ...]
    tree_index: Optional[int]
    parent_index: Optional[int] = None
    local_logprob: Optional[float] = None
    rank_among_siblings: Optional[int] = None
    sibling_margin: Optional[float] = None
    cumulative_path_logprob: Optional[float] = None
    sam_match_length: Optional[int] = None
    parent_top1_logit: Optional[float] = None
    parent_top2_logit: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OracleCandidate":
        source = str(data.get("source", ""))
        if source not in ("eagle", "sam"):
            raise ValueError("candidate source must be 'eagle' or 'sam': {}".format(source))
        path = int_tuple(data.get("path", []))
        token_path = int_tuple(data.get("token_path", []))
        if "token" in data:
            token = int(data["token"])
        elif token_path:
            token = int(token_path[-1])
        else:
            raise KeyError("token")
        if not path and len(token_path) > 1:
            path = token_path[:-1]
        depth = data.get("depth")
        if depth is None:
            if path:
                depth = len(path)
            elif token_path:
                depth = max(len(token_path) - 1, 1)
            else:
                depth = 1
        tree_index = data.get("tree_index", data.get("index", data.get("tree_idx")))
        return cls(
            source=source,
            token=token,
            depth=int(depth),
            score=float(data.get("score", 0.0)),
            accepted=bool_value(data.get("accepted", False)),
            path=path,
            token_path=token_path,
            tree_index=(int(tree_index) if tree_index is not None else None),
            parent_index=optional_int(data.get("parent_index")),
            local_logprob=optional_float(data.get("local_logprob")),
            rank_among_siblings=optional_int(data.get("rank_among_siblings")),
            sibling_margin=optional_float(data.get("sibling_margin")),
            cumulative_path_logprob=optional_float(data.get("cumulative_path_logprob")),
            sam_match_length=optional_int(data.get("sam_match_length")),
            parent_top1_logit=optional_float(data.get("parent_top1_logit")),
            parent_top2_logit=optional_float(data.get("parent_top2_logit")),
        )


@dataclass(frozen=True)
class DecodeStep:
    step: int
    candidates: List[OracleCandidate]
    acceptance_path: List[int]
    stats: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    first_rejected_depth: Optional[int] = None
    first_rejected_tree_index: Optional[int] = None
    first_rejected_parent_index: Optional[int] = None
    first_rejected_parent_path: Optional[Tuple[int, ...]] = None
    has_rejection_boundary_label: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any], fallback_step: int) -> "DecodeStep":
        candidates = [
            OracleCandidate.from_dict(item)
            for item in data.get("candidates", [])
        ]
        acceptance_path = [int(token) for token in data.get("acceptance_path", [])]
        stats = dict(data.get("stats", {}))
        label_fields_present = (
            "first_rejected_depth" in data
            or "first_rejected_depth" in stats
        )
        rejected_parent_path = data.get(
            "first_rejected_parent_path",
            stats.get("first_rejected_parent_path"),
        )
        metadata = dict(data.get("metadata", {}))
        for key in ("question_id", "question_index", "question_idx", "question_number"):
            if key in data and key not in metadata:
                metadata[key] = data[key]
        return cls(
            step=int(data.get("step", fallback_step)),
            candidates=candidates,
            acceptance_path=acceptance_path,
            stats=stats,
            metadata=metadata,
            first_rejected_depth=optional_int(
                data.get("first_rejected_depth", stats.get("first_rejected_depth"))
            ),
            first_rejected_tree_index=optional_int(
                data.get("first_rejected_tree_index", stats.get("first_rejected_tree_index"))
            ),
            first_rejected_parent_index=optional_int(
                data.get("first_rejected_parent_index", stats.get("first_rejected_parent_index"))
            ),
            first_rejected_parent_path=(
                int_tuple(rejected_parent_path)
                if rejected_parent_path is not None
                else None
            ),
            has_rejection_boundary_label=label_fields_present,
        )


@dataclass(frozen=True)
class MethodResult:
    method: str
    mat: float
    nodes: float
    mat_per_node: float
    steps: int
    oracle_gap: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "mat": self.mat,
            "nodes": self.nodes,
            "mat_per_node": self.mat_per_node,
            "steps": self.steps,
            "oracle_gap": self.oracle_gap,
        }
