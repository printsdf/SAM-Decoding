import os
import json
import torch
from dataclasses import dataclass, field
from numbers import Real
from typing import Optional, Union, List, Literal, Dict, Any
from enum import Enum


@dataclass
class SamdConfig:
    n_predicts: int = field(default=40)
    max_predicts: int = field(default=70)
    len_threshold: int = field(default=5)
    len_bias: int = field(default=5)

    cache_type: Literal["dynamic", "static"] = field(
        default="static"
    )
    use_last_hidden_states: bool = field(default=False)

    tree_method: Literal["token_recycle", "eagle", "eagle2", "eagle3"] = field(
        default="token_recycle"
    )
    tree_fusion: Literal[
        "none",
        "sam_sequence_graft",
        "sam_tree_union_prune",
        "eagle_prefix_sam_expand",
    ] = field(default="none")
    fusion_mode: Literal["none", "drafter_mars"] = field(default="none")
    fusion_max_draft_tokens: int = field(default=60)
    fusion_dedup_strategy: Literal["max_score", "sum_score", "keep_both"] = field(
        default="max_score"
    )
    fusion_truncate_strategy: Literal["score", "depth_first"] = field(default="score")
    drafter_mars_theta: float = field(default=0.90)
    drafter_mars_variant: Literal["top_path"] = field(default="top_path")
    drafter_mars_repair: Literal["graft", "naive_fuse"] = field(default="graft")
    drafter_mars_adaptive_theta: bool = field(default=False)
    drafter_mars_target_trigger_rate: float = field(default=0.75)
    drafter_mars_theta_step: float = field(default=0.02)
    drafter_mars_max_grafts: int = field(default=1)
    drafter_mars_extend: bool = field(default=False)
    sam_tree_max_nodes: int = field(default=16)
    sam_tree_top_k: int = field(default=4)
    sam_tree_alpha: float = field(default=4.0)
    sam_tree_max_depth: Optional[int] = field(default=6)
    sam_prefix_max_added_nodes: int = field(default=4)
    sam_prefix_top_k: int = field(default=2)
    sam_prefix_min_depth: int = field(default=1)
    sam_prefix_max_depth: Optional[int] = field(default=4)
    tree_model_path: Optional[str] = field(default=None)
    eagle3_total_token: int = field(default=60)
    eagle3_depth: int = field(default=7)
    eagle3_top_k: int = field(default=10)
    tree_path: Optional[str] = field(default=None)
    tree: Optional[List[List[int]]] = field(default=None)
    tree_config: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        if self.tree_fusion not in (
            "none",
            "sam_sequence_graft",
            "sam_tree_union_prune",
            "eagle_prefix_sam_expand",
        ):
            raise ValueError("unsupported tree_fusion: {}".format(self.tree_fusion))
        if self.tree_fusion in (
            "sam_sequence_graft",
            "sam_tree_union_prune",
            "eagle_prefix_sam_expand",
        ) and self.tree_method != "eagle3":
            raise ValueError(
                'tree_fusion="{}" only supports tree_method="eagle3"'.format(self.tree_fusion)
            )
        if self.fusion_mode not in ("none", "drafter_mars"):
            raise ValueError("unsupported fusion_mode: {}".format(self.fusion_mode))
        if not isinstance(self.drafter_mars_theta, Real) or isinstance(self.drafter_mars_theta, bool):
            raise ValueError("drafter_mars_theta must be a positive number")
        if self.drafter_mars_theta <= 0:
            raise ValueError("drafter_mars_theta must be a positive number")
        if self.drafter_mars_variant != "top_path":
            raise ValueError(
                "unsupported drafter_mars_variant: {}".format(self.drafter_mars_variant)
            )
        if self.drafter_mars_repair not in ("graft", "naive_fuse"):
            raise ValueError(
                "unsupported drafter_mars_repair: {}".format(self.drafter_mars_repair)
            )
        if not isinstance(self.drafter_mars_adaptive_theta, bool):
            raise ValueError("drafter_mars_adaptive_theta must be a bool")
        if not isinstance(self.drafter_mars_target_trigger_rate, Real) or isinstance(
            self.drafter_mars_target_trigger_rate, bool
        ):
            raise ValueError("drafter_mars_target_trigger_rate must be in (0, 1)")
        if not 0.0 < self.drafter_mars_target_trigger_rate < 1.0:
            raise ValueError("drafter_mars_target_trigger_rate must be in (0, 1)")
        if not isinstance(self.drafter_mars_theta_step, Real) or isinstance(
            self.drafter_mars_theta_step, bool
        ):
            raise ValueError("drafter_mars_theta_step must be a positive number")
        if self.drafter_mars_theta_step <= 0:
            raise ValueError("drafter_mars_theta_step must be a positive number")
        if not isinstance(self.drafter_mars_max_grafts, int) or isinstance(
            self.drafter_mars_max_grafts, bool
        ):
            raise ValueError("drafter_mars_max_grafts must be a positive integer")
        if self.drafter_mars_max_grafts < 1:
            raise ValueError("drafter_mars_max_grafts must be a positive integer")
        if not isinstance(self.drafter_mars_extend, bool):
            raise ValueError("drafter_mars_extend must be a bool")
        if self.fusion_mode == "drafter_mars":
            # Cache guard sizing: max_predicts must cover the largest per-step
            # draft. Repair (triggered) and leaf extension (not triggered) are
            # mutually exclusive per step; extension uses the SAM n_predicts
            # horizon like the baseline sequence draft (which itself needs
            # n_predicts + 1). The stock default 70 only covered tree + 8.
            repair_worst = self.drafter_mars_max_grafts * self.n_predicts
            extend_worst = self.n_predicts if self.drafter_mars_extend else 0
            worst_step = max(
                self.n_predicts + 1,
                self.eagle3_total_token + 1 + max(repair_worst, extend_worst),
            )
            self.max_predicts = max(self.max_predicts, worst_step + 2)
        if self.fusion_mode != "none" and self.tree_method != "eagle3":
            raise ValueError(
                'fusion_mode="{}" only supports tree_method="eagle3"'.format(self.fusion_mode)
            )
        if self.fusion_mode != "none" and self.tree_fusion != "none":
            raise ValueError("fusion_mode and tree_fusion cannot both be enabled")
        if not isinstance(self.fusion_max_draft_tokens, int) or isinstance(
            self.fusion_max_draft_tokens, bool
        ):
            raise ValueError("fusion_max_draft_tokens must be a positive integer")
        if self.fusion_max_draft_tokens <= 0:
            raise ValueError("fusion_max_draft_tokens must be a positive integer")
        if self.fusion_dedup_strategy not in ("max_score", "sum_score", "keep_both"):
            raise ValueError(
                "unsupported fusion_dedup_strategy: {}".format(self.fusion_dedup_strategy)
            )
        if self.fusion_truncate_strategy not in ("score", "depth_first"):
            raise ValueError(
                "unsupported fusion_truncate_strategy: {}".format(
                    self.fusion_truncate_strategy
                )
            )
        if not isinstance(self.eagle3_total_token, int) or isinstance(self.eagle3_total_token, bool):
            raise ValueError("eagle3_total_token must be a positive integer")
        if self.eagle3_total_token <= 0:
            raise ValueError("eagle3_total_token must be a positive integer")
        if not isinstance(self.eagle3_depth, int) or isinstance(self.eagle3_depth, bool):
            raise ValueError("eagle3_depth must be a positive integer")
        if self.eagle3_depth <= 0:
            raise ValueError("eagle3_depth must be a positive integer")
        if not isinstance(self.eagle3_top_k, int) or isinstance(self.eagle3_top_k, bool):
            raise ValueError("eagle3_top_k must be a positive integer")
        if self.eagle3_top_k <= 0:
            raise ValueError("eagle3_top_k must be a positive integer")
        if not isinstance(self.sam_tree_max_nodes, int) or isinstance(self.sam_tree_max_nodes, bool):
            raise ValueError("sam_tree_max_nodes must be a positive integer")
        if self.sam_tree_max_nodes <= 0:
            raise ValueError("sam_tree_max_nodes must be a positive integer")
        if not isinstance(self.sam_tree_top_k, int) or isinstance(self.sam_tree_top_k, bool):
            raise ValueError("sam_tree_top_k must be a positive integer")
        if self.sam_tree_top_k <= 0:
            raise ValueError("sam_tree_top_k must be a positive integer")
        if not isinstance(self.sam_tree_alpha, Real) or isinstance(self.sam_tree_alpha, bool):
            raise ValueError("sam_tree_alpha must be a positive number")
        if self.sam_tree_alpha <= 0:
            raise ValueError("sam_tree_alpha must be a positive number")
        if self.sam_tree_max_depth is not None:
            if not isinstance(self.sam_tree_max_depth, int) or isinstance(self.sam_tree_max_depth, bool):
                raise ValueError("sam_tree_max_depth must be None or a positive integer")
            if self.sam_tree_max_depth <= 0:
                raise ValueError("sam_tree_max_depth must be None or a positive integer")
        if not isinstance(self.sam_prefix_max_added_nodes, int) or isinstance(self.sam_prefix_max_added_nodes, bool):
            raise ValueError("sam_prefix_max_added_nodes must be a non-negative integer")
        if self.sam_prefix_max_added_nodes < 0:
            raise ValueError("sam_prefix_max_added_nodes must be a non-negative integer")
        if not isinstance(self.sam_prefix_top_k, int) or isinstance(self.sam_prefix_top_k, bool):
            raise ValueError("sam_prefix_top_k must be a positive integer")
        if self.sam_prefix_top_k <= 0:
            raise ValueError("sam_prefix_top_k must be a positive integer")
        if not isinstance(self.sam_prefix_min_depth, int) or isinstance(self.sam_prefix_min_depth, bool):
            raise ValueError("sam_prefix_min_depth must be a non-negative integer")
        if self.sam_prefix_min_depth < 0:
            raise ValueError("sam_prefix_min_depth must be a non-negative integer")
        if self.sam_prefix_max_depth is not None:
            if not isinstance(self.sam_prefix_max_depth, int) or isinstance(self.sam_prefix_max_depth, bool):
                raise ValueError("sam_prefix_max_depth must be None or a positive integer")
            if self.sam_prefix_max_depth <= 0:
                raise ValueError("sam_prefix_max_depth must be None or a positive integer")
            if self.sam_prefix_max_depth < self.sam_prefix_min_depth:
                raise ValueError("sam_prefix_max_depth must be >= sam_prefix_min_depth")
        if self.tree is None:
            if self.tree_method == "token_recycle":
                self.tree = load_token_recycle(self.tree_path)
            elif self.tree_method == "eagle":
                tree, tree_config = load_eagle(self.tree_model_path, self.tree_path)
                self.tree = tree
                self.tree_config = tree_config
                self.use_last_hidden_states = True
            elif self.tree_method == "eagle2":
                tree_config = load_eagle2(self.tree_model_path)
                self.tree_config = tree_config
                self.use_last_hidden_states = True
            elif self.tree_method == "eagle3":
                tree_config = load_eagle3(self.tree_model_path)
                self.tree_config = tree_config
                self.use_last_hidden_states = True
            else:
                raise ValueError

    @property
    def fusion_config(self) -> Optional[Any]:
        if self.fusion_mode == "none":
            return None
        from .fusion.types import FusionConfig

        return FusionConfig(
            mode="naive",
            max_draft_tokens=self.fusion_max_draft_tokens,
            dedup_strategy=self.fusion_dedup_strategy,
            truncate_strategy=self.fusion_truncate_strategy,
        )


class ForwardType(str, Enum):
    prefill = "prefill"
    seq_decode = "seq_decode"
    tree_decode = "tree_decode"


class ForwardState:

    def __init__(self, forward_type: ForwardType | None) -> None:
        self.forward_type = forward_type


class MaskState:

    def __init__(self, mask: Optional[torch.Tensor]) -> None:
        self.mask = mask

    def set_state(self, mask: Optional[torch.Tensor]) -> None:
        self.mask = mask


def load_token_recycle(tree_path: Optional[str] = None):
    if tree_path is None:
        tree_path = "token_recycle.json"
    samd_path = os.path.dirname(__file__)
    with open(os.path.join(samd_path, "config", tree_path), "r") as f:
        tree_adj: dict = json.load(f)["tree_adj"]
    num_node = len(tree_adj)
    tree: List[List[int]] = []
    for i in range(num_node):
        tree.append(tree_adj[str(i)])
    print("tree_path:", tree_path)
    print("len_tree:", len(tree))
    return tree


def load_eagle(tree_model_path: str, tree_path: Optional[str] = None):
    if tree_path is None:
        tree_path = "eagle.json"
    samd_path = os.path.dirname(__file__)
    with open(os.path.join(samd_path, "config", tree_path), "r") as f:
        tree = json.load(f)["tree_choices"]
    with open(os.path.join(tree_model_path, "config.json")) as f:
        tree_config = json.load(f)
    return tree, tree_config


def load_eagle2(tree_model_path: str):
    with open(os.path.join(tree_model_path, "config.json")) as f:
        tree_config = json.load(f)
    return tree_config


def load_eagle3(tree_model_path: str):
    with open(os.path.join(tree_model_path, "config.json")) as f:
        tree_config = json.load(f)
    return tree_config
