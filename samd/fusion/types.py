from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch


FusionMode = Literal["naive", "payoff_aware", "tree_aware"]
FusionDedupStrategy = Literal["max_score", "sum_score", "keep_both"]
FusionTruncateStrategy = Literal["score", "depth_first"]


@dataclass
class CandidateNode:
    """One non-root node in a fused draft tree."""

    token: int
    source: Literal["eagle", "sam", "both"]
    score: float
    depth: int
    path: List[int]
    parent_idx: Optional[int] = None
    normalized_score: float = 0.0
    source_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return tuple(self.path), int(self.token)

    @property
    def token_path(self) -> tuple:
        return tuple(self.path + [int(self.token)])


@dataclass
class FusedTree:
    """Fused candidate tree plus verifier-compatible buffers."""

    nodes: List[CandidateNode]
    tokens: torch.Tensor
    tree_mask: torch.Tensor
    position_ids: torch.Tensor
    retrieve_indices: torch.Tensor
    buffers_kwargs: Dict[str, torch.Tensor]
    metadata: Dict


@dataclass
class FusionConfig:
    """Configuration for dual-draft fusion strategies."""

    mode: FusionMode = "naive"
    max_draft_tokens: int = 60
    dedup_strategy: FusionDedupStrategy = "max_score"
    truncate_strategy: FusionTruncateStrategy = "score"
    debug: bool = False
    metadata: Dict = field(default_factory=dict)
