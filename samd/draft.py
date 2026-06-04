import torch
from typing import List, Tuple, Dict, Optional
from enum import Enum
from collections import namedtuple

from .samd_config import SamdConfig
from .sam import DynSAM, StaticSAM, NullStaticSAM
from .sam.prefix_expansion import (
    SamPrefixBudget,
    expand_local_siblings,
    find_prefix_anchors,
)
from .sam.tree_draft import SamTreeBudget
from .tree_model import TreeModel, tree_model_cls
from .tree_model.fusion import TreeSpec
from transformers import LlamaConfig, LlamaForCausalLM

from profile_utils import profile_decorator, profile_lookup_decorator

# from transformers import LlamaTokenizer
# tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained('/data/models/vicuna-7b-v1.3')

class CandidateType(str, Enum):
    sequence = "sequence"
    tree = "tree"

Candidates = namedtuple('Candidates', ['type', 'tokens', 'candidate_tokens', 'buffers_kwargs'])

TOPK = 8

class DraftModel(torch.nn.Module):
    
    def __init__(self,
        config: SamdConfig,
        sam_dyn: DynSAM = None,
        sam_static: StaticSAM = None,
        tree_model: TreeModel = None,
        lm: LlamaForCausalLM = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        tree_cls = tree_model_cls[config.tree_method]
        self.config = config
        self.sam_dyn = sam_dyn if sam_dyn is not None else DynSAM(config.n_predicts)
        self.sam_static = sam_static if sam_static is not None else NullStaticSAM(config.n_predicts)
        self.tree_model = tree_model if tree_model is not None else tree_cls(config, lm, dtype, device)
        
        self.sam_dyn.n_predicts = config.n_predicts
        self.sam_static.n_predicts = config.n_predicts
        self.len_bias = config.len_bias
        self.len_threshold = config.len_threshold
        self.sam_tree_budget = SamTreeBudget.from_config(config)
        self.sam_prefix_budget = SamPrefixBudget.from_config(config)
        self.reset_fusion_stats()

    def reset_fusion_stats(self):
        mode = self.config.tree_fusion
        budget = (
            self.sam_tree_budget.to_dict()
            if mode == "sam_tree_union_prune"
            else self.sam_prefix_budget.to_dict()
        )
        self.fusion_stats = {
            "enabled": mode in ("sam_tree_union_prune", "eagle_prefix_sam_expand"),
            "mode": mode,
            "budget": budget,
            "steps": 0,
            "matched_steps": 0,
            "pure_eagle3_steps": 0,
            "budget_hit_steps": 0,
            "node_count_sum": 0,
            "sam_added_nodes_sum": 0,
            "merged_nodes_sum": 0,
            "anchor_count_sum": 0,
            "added_nodes_sum": 0,
            "retained_leaf_count_sum": 0,
            "missing_leaf_count_sum": 0,
            "last": None,
        }

    def fusion_summary(self):
        if not self.fusion_stats.get("enabled"):
            return None
        steps = max(self.fusion_stats["steps"], 1)
        summary = {
            "mode": self.fusion_stats["mode"],
            "budget": self.fusion_stats["budget"],
            "steps": self.fusion_stats["steps"],
            "matched_steps": self.fusion_stats["matched_steps"],
            "pure_eagle3_steps": self.fusion_stats["pure_eagle3_steps"],
            "avg_node_count": self.fusion_stats["node_count_sum"] / steps,
            "last": self.fusion_stats["last"],
        }
        if self.fusion_stats["mode"] == "sam_tree_union_prune":
            summary.update({
                "budget_hit_steps": self.fusion_stats["budget_hit_steps"],
                "avg_sam_added_nodes": self.fusion_stats["sam_added_nodes_sum"] / steps,
                "avg_merged_nodes": self.fusion_stats["merged_nodes_sum"] / steps,
            })
        if self.fusion_stats["mode"] == "eagle_prefix_sam_expand":
            summary.update({
                "avg_anchor_count": self.fusion_stats["anchor_count_sum"] / steps,
                "avg_added_nodes": self.fusion_stats["added_nodes_sum"] / steps,
                "avg_retained_leaf_count": self.fusion_stats["retained_leaf_count_sum"] / steps,
                "avg_missing_leaf_count": self.fusion_stats["missing_leaf_count_sum"] / steps,
            })
        return summary
        
    def reset(self):
        self.sam_dyn.reset()
        self.sam_static.reset()
        self.tree_model.reset()
        self.reset_fusion_stats()

    def lookup(self, start_token: int):
        index_dyn, match_dyn = self.sam_dyn.lookup(start_token)
        index_static, match_static = self.sam_static.lookup(start_token)
        match_static -= self.len_bias

        if self.config.tree_fusion == "sam_sequence_graft":
            pred_ids, buffers_kwargs = self.tree_model.gen_draft(start_token)
            if max(match_dyn, match_static) < self.len_threshold:
                return CandidateType.tree, pred_ids, buffers_kwargs

            if match_dyn >= match_static:
                seq = self.sam_dyn.gen_draft_raw(index_dyn, start_token, self.config.n_predicts)
            else:
                seq = self.sam_static.gen_draft_raw(index_static, start_token, self.config.n_predicts)

            tree_spec = TreeSpec.from_eagle3_buffers(
                pred_ids,
                buffers_kwargs["tree_attn_mask"],
                buffers_kwargs["tree_position_ids"],
            ).graft_sequence(seq)
            tree_attn_mask = buffers_kwargs["tree_attn_mask"]
            tree_position_ids = buffers_kwargs["tree_position_ids"]
            fused_kwargs = tree_spec.to_buffers(
                device=tree_position_ids.device,
                mask_dtype=tree_attn_mask.dtype,
            )
            return CandidateType.tree, list(tree_spec.tokens), fused_kwargs

        if self.config.tree_fusion == "eagle_prefix_sam_expand":
            pred_ids, buffers_kwargs = self.tree_model.gen_draft(start_token)
            self.fusion_stats["steps"] += 1
            sam_hit = max(match_dyn, match_static) >= self.len_threshold
            dynamic_hit = match_dyn >= self.len_threshold
            if not sam_hit:
                node_count = len(pred_ids)
                self.fusion_stats["pure_eagle3_steps"] += 1
                self.fusion_stats["node_count_sum"] += node_count
                self.fusion_stats["last"] = {
                    "match_length": int(match_dyn),
                    "match_static": int(match_static),
                    "node_count": node_count,
                    "anchor_count": 0,
                    "added_nodes": 0,
                    "retained_leaf_count": 0,
                    "missing_leaf_count": 0,
                    "hit": False,
                }
                return CandidateType.tree, pred_ids, buffers_kwargs

            if match_dyn >= match_static:
                seq = self.sam_dyn.gen_draft_raw(index_dyn, start_token, self.config.n_predicts)
            else:
                seq = self.sam_static.gen_draft_raw(index_static, start_token, self.config.n_predicts)

            baseline_spec = TreeSpec.from_eagle3_buffers(
                pred_ids,
                buffers_kwargs["tree_attn_mask"],
                buffers_kwargs["tree_position_ids"],
            ).graft_sequence(seq)
            fused_spec = baseline_spec
            anchor_stats = {
                "anchor_count": 0,
                "mapped_node_count": 0,
                "unmapped_node_count": 0,
                "skipped_leaf_anchor_count": 0,
                "skipped_depth_count": 0,
            }
            expand_stats = {
                "added_nodes": 0,
                "expanded_anchor_count": 0,
                "skipped_duplicate_sibling_count": 0,
                "skipped_leaf_anchor_count": 0,
                "skipped_budget_sibling_count": 0,
                "anchor_without_next_count": 0,
                "pruned_by_top_k": 0,
            }
            if self.sam_prefix_budget.max_added_nodes > 0 and dynamic_hit:
                anchor_start_index = self.sam_dyn.to_anc(index_dyn)
                anchor_start_length = min(
                    int(match_dyn),
                    int(self.sam_dyn.states[anchor_start_index].length),
                )
                anchors, anchor_stats = find_prefix_anchors(
                    baseline_spec,
                    self.sam_dyn.states,
                    anchor_start_index,
                    anchor_start_length,
                    self.sam_prefix_budget,
                )
                fused_spec, expand_stats = expand_local_siblings(
                    baseline_spec,
                    anchors,
                    self.sam_dyn.states,
                    self.sam_prefix_budget,
                )
                fused_spec.assert_leaf_paths_retained(
                    baseline_spec,
                    context="eagle_prefix_sam_expand leaf retention",
                )
            leaf_stats = fused_spec.leaf_retention_stats(baseline_spec)
            self.fusion_stats["matched_steps"] += 1
            self.fusion_stats["node_count_sum"] += len(fused_spec.tokens)
            self.fusion_stats["anchor_count_sum"] += anchor_stats["anchor_count"]
            self.fusion_stats["added_nodes_sum"] += expand_stats["added_nodes"]
            self.fusion_stats["retained_leaf_count_sum"] += leaf_stats["retained_leaf_count"]
            self.fusion_stats["missing_leaf_count_sum"] += leaf_stats["missing_leaf_count"]
            self.fusion_stats["last"] = {
                "match_length": int(match_dyn),
                "match_static": int(match_static),
                "node_count": len(fused_spec.tokens),
                "baseline_node_count": len(baseline_spec.tokens),
                "anchor_count": anchor_stats["anchor_count"],
                "added_nodes": expand_stats["added_nodes"],
                "retained_leaf_count": leaf_stats["retained_leaf_count"],
                "missing_leaf_count": leaf_stats["missing_leaf_count"],
                "hit": True,
                "dynamic_hit": bool(dynamic_hit),
                "anchor_stats": anchor_stats,
                "expand_stats": expand_stats,
            }

            tree_attn_mask = buffers_kwargs["tree_attn_mask"]
            tree_position_ids = buffers_kwargs["tree_position_ids"]
            fused_kwargs = fused_spec.to_buffers(
                device=tree_position_ids.device,
                mask_dtype=tree_attn_mask.dtype,
            )
            return CandidateType.tree, list(fused_spec.tokens), fused_kwargs

        if self.config.tree_fusion == "sam_tree_union_prune":
            pred_ids, buffers_kwargs = self.tree_model.gen_draft(start_token)
            self.fusion_stats["steps"] += 1
            if match_dyn < self.len_threshold:
                node_count = len(pred_ids)
                self.fusion_stats["pure_eagle3_steps"] += 1
                self.fusion_stats["node_count_sum"] += node_count
                self.fusion_stats["last"] = {
                    "match_length": int(match_dyn),
                    "node_count": node_count,
                    "sam_added_nodes": 0,
                    "merged_nodes": 0,
                    "budget_hit": False,
                }
                return CandidateType.tree, pred_ids, buffers_kwargs

            sam_tree, sam_stats = self.sam_dyn.gen_tree_draft(
                index_dyn,
                match_dyn,
                start_token,
                self.sam_tree_budget,
            )
            tree_spec = TreeSpec.from_eagle3_buffers(
                pred_ids,
                buffers_kwargs["tree_attn_mask"],
                buffers_kwargs["tree_position_ids"],
            )
            fused_spec, union_stats = tree_spec.union_sam_tree(sam_tree)
            tree_attn_mask = buffers_kwargs["tree_attn_mask"]
            tree_position_ids = buffers_kwargs["tree_position_ids"]
            fused_kwargs = fused_spec.to_buffers(
                device=tree_position_ids.device,
                mask_dtype=tree_attn_mask.dtype,
            )

            budget_hit = bool(sam_stats["budget_hit"])
            self.fusion_stats["matched_steps"] += 1
            self.fusion_stats["node_count_sum"] += union_stats["node_count"]
            self.fusion_stats["sam_added_nodes_sum"] += union_stats["sam_added_nodes"]
            self.fusion_stats["merged_nodes_sum"] += union_stats["merged_nodes"]
            if budget_hit:
                self.fusion_stats["budget_hit_steps"] += 1
            self.fusion_stats["last"] = {
                "match_length": int(match_dyn),
                "node_count": union_stats["node_count"],
                "sam_added_nodes": union_stats["sam_added_nodes"],
                "merged_nodes": union_stats["merged_nodes"],
                "budget_hit": budget_hit,
                "sam_target_nodes": sam_stats["target_nodes"],
                "sam_node_count": sam_stats["node_count"],
            }
            return CandidateType.tree, list(fused_spec.tokens), fused_kwargs

        if max(match_dyn, match_static) >= self.len_threshold:
            if match_dyn >= match_static:
                seq = self.sam_dyn.gen_draft(index_dyn, start_token)
            else:
                seq = self.sam_static.gen_draft(index_static, start_token)
            return (CandidateType.sequence, seq, {})
        else:
            return (CandidateType.tree,) + self.tree_model.gen_draft(start_token)
    
    def update(self,
        tokens: Optional[torch.Tensor] = None,
        last_hidden_states: Optional[torch.Tensor] = None,
        tree_tokens: Optional[torch.Tensor] = None,
        tree_logits: Optional[torch.Tensor] = None,
    ):
        tokens_list = tokens.tolist()
        self.sam_dyn.add_tokens(tokens_list)
        self.sam_static.transfer_tokens(tokens_list)
        self.tree_model.update(
            tokens=tokens,
            last_hidden_states=last_hidden_states,
            tree_tokens=tree_tokens, 
            tree_logits=tree_logits,
        )
