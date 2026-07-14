import torch
import torch.nn.functional as F
import random
from enum import Enum
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import namedtuple
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from profile_utils import profile_decorator
from .samd_config import SamdConfig
from .draft import DraftModel, Candidates, CandidateType    
from .fusion.naive_fusion import (
    candidate_trace_records,
    fuse_eagle_sam_naive,
    parse_eagle_tree,
)


def _extract_candidate_tokens_eagle(pred_ids, buffers, device: torch.device) -> torch.Tensor:
    retrieve_indices = buffers.get("tree_retrieve_indices")
    if isinstance(pred_ids, torch.Tensor):
        tokens = pred_ids.to(device=device, dtype=torch.long).view(-1)
    else:
        tokens = torch.tensor(pred_ids, dtype=torch.long, device=device)
    tokens_ext = torch.cat([tokens, tokens.new_zeros(1)])
    if retrieve_indices is None:
        return tokens.unsqueeze(0)
    if isinstance(retrieve_indices, torch.Tensor):
        retrieve_indices = retrieve_indices.to(device=device, dtype=torch.long)
    else:
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long, device=device)
    return tokens_ext[retrieve_indices]


class OptionalTensor:
    
    def __init__(self, data: Optional[torch.Tensor] = None):
        self.data = data

    def apply(self, fn: Callable) -> 'OptionalTensor':
        if self.data is None:
            return OptionalTensor(None)
        else:
            return OptionalTensor(fn(self.data))

@dataclass
class SamdGenerationConfig:
    max_steps: int = field(default=512)
    max_new_tokens: int = field(default=512)
    max_cache_len: int = field(default=2048)
    greedy: bool = field(default=True)
    temperature: float = field(default=0.0)
    top_p: float = field(default=0.0)
    top_k: int = field(default=0)
    logits_processor: LogitsProcessorList = field(default=None)
    collect_diagnosis_trace: bool = field(default=False)
    fusion_profiler: Optional[Any] = field(default=None)
    
    def __post_init__(self):
        if not self.greedy:
            assert self.temperature >= 1e-5
            self.logits_processor = self.prepare_logits_processor(
                self.temperature,
                self.top_p,
                self.top_k,
            )

    @staticmethod
    def prepare_logits_processor(
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
    ) -> LogitsProcessorList:
        processor_list = LogitsProcessorList()
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list


def _active_fusion_profiler(gen_config: SamdGenerationConfig):
    profiler = getattr(gen_config, "fusion_profiler", None)
    if profiler is not None and getattr(profiler, "enabled", False):
        return profiler
    return None


@profile_decorator("gen_candidates")
def gen_candidates(
    sample_p: torch.Tensor,
    tree_retrieve_indices: torch.Tensor,
    draft: DraftModel,
    samd_config: SamdConfig,
    gen_config: SamdGenerationConfig,
    device: torch.device,
):
    """
    Generate candidates based on provided logits and indices.
    
    Parameters:
    - ...

    Returns:
    - tuple (torch.Tensor, List[int]): ...
    """
    # Greedy decoding: Select the most probable candidate from the original logits.
    if gen_config.greedy:
        start_token = torch.argmax(sample_p, dim=-1).item()
    else:
        start_token = torch.multinomial(sample_p, 1).item()

    profiler = _active_fusion_profiler(gen_config)

    if samd_config.fusion_mode == "naive":
        timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
        try:
            index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
            index_static, match_static_raw = draft.sam_static.lookup(start_token)
            match_static = match_static_raw - draft.len_bias
        finally:
            if profiler is not None:
                profiler.end_section("draft_sam", timer)
        best_match = max(match_dyn, match_static)
        samd_len_threshold = getattr(samd_config, "samd_len_threshold", None)
        if samd_len_threshold is None:
            samd_len_threshold = samd_config.len_threshold

        if best_match < samd_len_threshold:
            timer = profiler.start_section("draft_eagle") if profiler is not None else 0.0
            try:
                if profiler is not None:
                    eagle_pred_ids, eagle_buffers, eagle_logprobs, eagle_raw_logits = (
                        draft.tree_model.gen_draft(
                            start_token,
                            return_logprobs=True,
                            return_raw_logits=True,
                        )
                    )
                else:
                    eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
                    eagle_logprobs = None
                    eagle_raw_logits = None
            finally:
                if profiler is not None:
                    profiler.end_section("draft_eagle", timer)
            eagle_nonroot_nodes = max(len(eagle_pred_ids) - 1, 0)
            metadata = {
                "mode": "naive",
                "eagle_nodes": eagle_nonroot_nodes,
                "sam_nodes": 0,
                "merged_nodes": eagle_nonroot_nodes,
                "final_nodes": eagle_nonroot_nodes,
                "dedup_count": 0,
                "both_proposed_count": 0,
                "truncate_count": 0,
                "truncated_count": 0,
                "sam_skipped": True,
                "sam_match_quality": int(best_match),
                "threshold": samd_len_threshold,
                "sam_contribution": 0,
                "eagle_contribution": eagle_nonroot_nodes,
                "both_contribution": 0,
                "sam_avg_match_length": float(best_match),
                "sam_max_match_length": max(int(best_match), 0),
                "selected_eagle": eagle_nonroot_nodes,
                "selected_sam": 0,
                "selected_both": 0,
                "node_sources": ["root"] + ["eagle"] * eagle_nonroot_nodes,
                "sam_match_length": max(int(best_match), 0),
            }
            if profiler is not None:
                eagle_tree = {
                    "tokens": eagle_pred_ids,
                    "tree_attn_mask": eagle_buffers["tree_attn_mask"],
                    "tree_position_ids": eagle_buffers["tree_position_ids"],
                    "tree_retrieve_indices": eagle_buffers["tree_retrieve_indices"],
                }
                metadata["oracle_candidates"] = candidate_trace_records(
                    parse_eagle_tree(
                        eagle_tree,
                        start_token,
                        eagle_logprobs=eagle_logprobs,
                        eagle_raw_logits=eagle_raw_logits,
                    )
                )
            draft.record_naive_fusion(metadata)
            if profiler is not None:
                profiler.add_step_metadata({
                    "sam_skipped": True,
                    "sam_match_quality": int(best_match),
                    "threshold": int(samd_len_threshold),
                    "eagle_nodes": eagle_nonroot_nodes,
                    "sam_nodes": 0,
                    "final_nodes": eagle_nonroot_nodes,
                })
            return Candidates(
                type=CandidateType.tree,
                tokens=torch.tensor([eagle_pred_ids], dtype=torch.long, device=device),
                candidate_tokens=_extract_candidate_tokens_eagle(
                    eagle_pred_ids,
                    eagle_buffers,
                    device,
                ),
                buffers_kwargs=eagle_buffers,
            )

        timer = profiler.start_section("draft_eagle") if profiler is not None else 0.0
        try:
            if profiler is not None:
                eagle_tokens, eagle_buffers, eagle_logprobs, eagle_raw_logits = (
                    draft.tree_model.gen_draft(
                        start_token,
                        return_logprobs=True,
                        return_raw_logits=True,
                    )
                )
            else:
                eagle_tokens, eagle_buffers, eagle_logprobs = draft.tree_model.gen_draft(
                    start_token,
                    return_logprobs=True,
                )
                eagle_raw_logits = None
        finally:
            if profiler is not None:
                profiler.end_section("draft_eagle", timer)
        eagle_tree = {
            "tokens": eagle_tokens,
            "logprobs": eagle_logprobs,
            "raw_logits": eagle_raw_logits,
            "tree_attn_mask": eagle_buffers["tree_attn_mask"],
            "tree_position_ids": eagle_buffers["tree_position_ids"],
            "tree_retrieve_indices": eagle_buffers["tree_retrieve_indices"],
        }

        timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
        try:
            if match_dyn >= match_static:
                sam_candidates = draft.sam_dyn.gen_draft_raw(
                    index_dyn,
                    start_token,
                    samd_config.n_predicts,
                )
                sam_match_length = match_dyn
            else:
                sam_candidates = draft.sam_static.gen_draft_raw(
                    index_static,
                    start_token,
                    samd_config.n_predicts,
                )
                sam_match_length = max(match_static, 0)
        finally:
            if profiler is not None:
                profiler.end_section("draft_sam", timer)

        timer = profiler.start_section("fusion_logic") if profiler is not None else 0.0
        try:
            fusion_config = samd_config.fusion_config
            if profiler is not None:
                fusion_config.debug = True
            fused_tree = fuse_eagle_sam_naive(
                eagle_tree=eagle_tree,
                sam_candidates=sam_candidates,
                start_token=start_token,
                config=fusion_config,
                sam_match_length=sam_match_length,
                eagle_logprobs=eagle_logprobs,
                eagle_raw_logits=eagle_raw_logits,
            )
            fused_tree.metadata.setdefault("sam_skipped", False)
            draft.record_naive_fusion(fused_tree.metadata)
            tokens = fused_tree.tokens.unsqueeze(0)
            tokens_ext = torch.cat(
                [fused_tree.tokens, torch.zeros(1, dtype=torch.long, device=fused_tree.tokens.device)]
            )
            candidate_tokens = tokens_ext[fused_tree.retrieve_indices]
        finally:
            if profiler is not None:
                profiler.end_section("fusion_logic", timer)
        if profiler is not None:
            profiler.add_step_metadata({
                "sam_skipped": False,
                "sam_match_length": int(sam_match_length),
                "eagle_nodes": int(fused_tree.metadata.get("eagle_nodes", 0)),
                "sam_nodes": int(fused_tree.metadata.get("sam_nodes", 0)),
                "final_nodes": int(fused_tree.metadata.get("final_nodes", 0)),
                "selected_eagle": int(fused_tree.metadata.get("selected_eagle", 0)),
                "selected_sam": int(fused_tree.metadata.get("selected_sam", 0)),
                "selected_both": int(fused_tree.metadata.get("selected_both", 0)),
            })
        return Candidates(
            CandidateType.tree,
            tokens,
            candidate_tokens,
            fused_tree.buffers_kwargs,
        )

    if samd_config.fusion_mode == "drafter_mars":
        from .fusion.drafter_mars_gate import top_path_ratio_trigger
        from .tree_model.fusion import TreeSpec

        theta = samd_config.drafter_mars_theta
        timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
        try:
            index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
            index_static, match_static_raw = draft.sam_static.lookup(start_token)
            match_static = match_static_raw - draft.len_bias
        finally:
            if profiler is not None:
                profiler.end_section("draft_sam", timer)
        best_match = max(match_dyn, match_static)
        samd_len_threshold = getattr(samd_config, "samd_len_threshold", None)
        if samd_len_threshold is None:
            samd_len_threshold = samd_config.len_threshold

        if best_match >= samd_len_threshold:
            # Baseline parity: match-length hit always takes the SAM sequence draft.
            timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
            try:
                if match_dyn >= match_static:
                    seq = draft.sam_dyn.gen_draft(index_dyn, start_token)
                else:
                    seq = draft.sam_static.gen_draft(index_static, start_token)
            finally:
                if profiler is not None:
                    profiler.end_section("draft_sam", timer)
            sam_nonroot_nodes = max(len(seq) - 1, 0)
            metadata = {
                "mode": "drafter_mars",
                "ratio_triggered": False,
                "max_top_path_ratio": None,
                "drafter_mars_theta": float(theta),
                "eagle_nodes": 0,
                "sam_nodes": sam_nonroot_nodes,
                "final_nodes": sam_nonroot_nodes,
                "sam_skipped": False,
                "sam_match_length": int(best_match),
                "threshold": int(samd_len_threshold),
                "node_sources": ["root"] + ["sam"] * sam_nonroot_nodes,
            }
            draft.record_naive_fusion(metadata)
            if profiler is not None:
                profiler.add_step_metadata(metadata)
            tokens = torch.tensor([seq], dtype=torch.long, device=device)
            return Candidates(CandidateType.sequence, tokens, tokens, {})

        timer = profiler.start_section("draft_eagle") if profiler is not None else 0.0
        try:
            eagle_tokens, eagle_buffers, eagle_logprobs, eagle_raw_logits = (
                draft.tree_model.gen_draft(
                    start_token,
                    return_logprobs=True,
                    return_raw_logits=True,
                )
            )
        finally:
            if profiler is not None:
                profiler.end_section("draft_eagle", timer)
        eagle_nonroot_nodes = max(len(eagle_tokens) - 1, 0)

        timer = profiler.start_section("fusion_logic") if profiler is not None else 0.0
        try:
            tree_spec = TreeSpec.from_eagle3_buffers(
                eagle_tokens,
                eagle_buffers["tree_attn_mask"],
                eagle_buffers["tree_position_ids"],
            )
            ratio_triggered, max_top_path_ratio = top_path_ratio_trigger(
                tree_spec.parents,
                eagle_logprobs,
                eagle_raw_logits,
                theta,
            )
        finally:
            if profiler is not None:
                profiler.end_section("fusion_logic", timer)

        sam_candidates = None
        sam_match_length = 0
        if ratio_triggered:
            timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
            try:
                if match_dyn >= match_static:
                    sam_candidates = draft.sam_dyn.gen_draft_raw(
                        index_dyn,
                        start_token,
                        samd_config.n_predicts,
                    )
                    sam_match_length = match_dyn
                else:
                    sam_candidates = draft.sam_static.gen_draft_raw(
                        index_static,
                        start_token,
                        samd_config.n_predicts,
                    )
                    sam_match_length = max(match_static, 0)
            finally:
                if profiler is not None:
                    profiler.end_section("draft_sam", timer)

        if ratio_triggered and sam_candidates is not None and len(sam_candidates) > 1:
            timer = profiler.start_section("fusion_logic") if profiler is not None else 0.0
            try:
                fusion_config = samd_config.fusion_config
                if profiler is not None:
                    fusion_config.debug = True
                fused_tree = fuse_eagle_sam_naive(
                    eagle_tree={
                        "tokens": eagle_tokens,
                        "logprobs": eagle_logprobs,
                        "raw_logits": eagle_raw_logits,
                        "tree_attn_mask": eagle_buffers["tree_attn_mask"],
                        "tree_position_ids": eagle_buffers["tree_position_ids"],
                        "tree_retrieve_indices": eagle_buffers["tree_retrieve_indices"],
                    },
                    sam_candidates=sam_candidates,
                    start_token=start_token,
                    config=fusion_config,
                    sam_match_length=sam_match_length,
                    eagle_logprobs=eagle_logprobs,
                    eagle_raw_logits=eagle_raw_logits,
                )
                fused_tree.metadata["mode"] = "drafter_mars"
                fused_tree.metadata["ratio_triggered"] = True
                fused_tree.metadata["max_top_path_ratio"] = float(max_top_path_ratio)
                fused_tree.metadata["drafter_mars_theta"] = float(theta)
                fused_tree.metadata.setdefault("sam_skipped", False)
                draft.record_naive_fusion(fused_tree.metadata)
                tokens = fused_tree.tokens.unsqueeze(0)
                tokens_ext = torch.cat(
                    [
                        fused_tree.tokens,
                        torch.zeros(1, dtype=torch.long, device=fused_tree.tokens.device),
                    ]
                )
                candidate_tokens = tokens_ext[fused_tree.retrieve_indices]
            finally:
                if profiler is not None:
                    profiler.end_section("fusion_logic", timer)
            if profiler is not None:
                profiler.add_step_metadata({
                    "mode": "drafter_mars",
                    "ratio_triggered": True,
                    "max_top_path_ratio": float(max_top_path_ratio),
                    "drafter_mars_theta": float(theta),
                    "sam_skipped": False,
                    "sam_match_length": int(sam_match_length),
                    "eagle_nodes": int(fused_tree.metadata.get("eagle_nodes", 0)),
                    "sam_nodes": int(fused_tree.metadata.get("sam_nodes", 0)),
                    "final_nodes": int(fused_tree.metadata.get("final_nodes", 0)),
                })
            return Candidates(
                CandidateType.tree,
                tokens,
                candidate_tokens,
                fused_tree.buffers_kwargs,
            )

        # Not triggered (or SAM has no continuation): pure EAGLE3 tree.
        metadata = {
            "mode": "drafter_mars",
            "ratio_triggered": bool(ratio_triggered),
            "max_top_path_ratio": (
                float(max_top_path_ratio) if max_top_path_ratio is not None else None
            ),
            "drafter_mars_theta": float(theta),
            "eagle_nodes": eagle_nonroot_nodes,
            "sam_nodes": 0,
            "final_nodes": eagle_nonroot_nodes,
            "selected_eagle": eagle_nonroot_nodes,
            "selected_sam": 0,
            "selected_both": 0,
            "node_sources": ["root"] + ["eagle"] * eagle_nonroot_nodes,
            "sam_skipped": True,
        }
        draft.record_naive_fusion(metadata)
        if profiler is not None:
            profiler.add_step_metadata(metadata)
        return Candidates(
            type=CandidateType.tree,
            tokens=torch.tensor([eagle_tokens], dtype=torch.long, device=device),
            candidate_tokens=_extract_candidate_tokens_eagle(
                eagle_tokens,
                eagle_buffers,
                device,
            ),
            buffers_kwargs=eagle_buffers,
        )

    if samd_config.fusion_mode == "rejection_boundary":
        from .fusion.rejection_boundary import (
            compute_confidence_margin,
            should_trigger_sam,
        )

        timer = profiler.start_section("draft_eagle") if profiler is not None else 0.0
        try:
            eagle_tokens, eagle_buffers, eagle_logprobs = draft.tree_model.gen_draft(
                start_token,
                return_logprobs=True,
            )
        finally:
            if profiler is not None:
                profiler.end_section("draft_eagle", timer)

        eagle_tree = {
            "tokens": eagle_tokens,
            "logprobs": eagle_logprobs,
            "tree_attn_mask": eagle_buffers["tree_attn_mask"],
            "tree_position_ids": eagle_buffers["tree_position_ids"],
            "tree_retrieve_indices": eagle_buffers["tree_retrieve_indices"],
        }

        # Extract root-level logprobs for confidence computation
        logprobs_tensor = torch.as_tensor(
            eagle_logprobs,
            dtype=torch.float32,
            device=device,
        ).view(-1)
        position_ids = eagle_buffers.get("tree_position_ids")
        root_logprobs = logprobs_tensor.new_empty(0)
        if position_ids is not None and logprobs_tensor.numel() > 0:
            if isinstance(position_ids, torch.Tensor):
                position_ids_tensor = position_ids.to(
                    device=device,
                    dtype=torch.long,
                ).view(-1)
            else:
                position_ids_tensor = torch.tensor(
                    position_ids,
                    dtype=torch.long,
                    device=device,
                ).view(-1)
            if position_ids_tensor.shape[0] == logprobs_tensor.shape[0]:
                root_logprobs = logprobs_tensor[position_ids_tensor == 1]
        if root_logprobs.numel() == 0 and logprobs_tensor.numel() > 1:
            root_logprobs = logprobs_tensor[1:]

        root_margin = compute_confidence_margin(root_logprobs.unsqueeze(0), depth=0)
        threshold = samd_config.rejection_conf_threshold
        eagle_nonroot_nodes = max(len(eagle_tokens) - 1, 0)

        if should_trigger_sam(root_margin, threshold):
            # Low confidence - trigger SAM
            timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
            try:
                index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
                index_static, match_static_raw = draft.sam_static.lookup(start_token)
                match_static = match_static_raw - draft.len_bias
                if match_dyn >= match_static:
                    sam_candidates = draft.sam_dyn.gen_draft_raw(
                        index_dyn,
                        start_token,
                        samd_config.n_predicts,
                    )
                    sam_match_length = match_dyn
                else:
                    sam_candidates = draft.sam_static.gen_draft_raw(
                        index_static,
                        start_token,
                        samd_config.n_predicts,
                    )
                    sam_match_length = max(match_static, 0)
            finally:
                if profiler is not None:
                    profiler.end_section("draft_sam", timer)

            timer = profiler.start_section("fusion_logic") if profiler is not None else 0.0
            try:
                fusion_config = samd_config.fusion_config
                if profiler is not None:
                    fusion_config.debug = True
                fused_tree = fuse_eagle_sam_naive(
                    eagle_tree=eagle_tree,
                    sam_candidates=sam_candidates,
                    start_token=start_token,
                    config=fusion_config,
                    sam_match_length=sam_match_length,
                    eagle_logprobs=eagle_logprobs,
                )
                fused_tree.metadata["mode"] = "rejection_boundary"
                fused_tree.metadata["root_margin"] = float(root_margin)
                fused_tree.metadata["rejection_conf_threshold"] = float(threshold)
                fused_tree.metadata.setdefault("sam_skipped", False)
                draft.record_naive_fusion(fused_tree.metadata)
                tokens = fused_tree.tokens.unsqueeze(0)
                tokens_ext = torch.cat(
                    [
                        fused_tree.tokens,
                        torch.zeros(
                            1,
                            dtype=torch.long,
                            device=fused_tree.tokens.device,
                        ),
                    ]
                )
                candidate_tokens = tokens_ext[fused_tree.retrieve_indices]
            finally:
                if profiler is not None:
                    profiler.end_section("fusion_logic", timer)
            if profiler is not None:
                profiler.add_step_metadata({
                    "sam_skipped": False,
                    "root_margin": float(root_margin),
                    "rejection_conf_threshold": float(threshold),
                    "eagle_nodes": int(fused_tree.metadata.get("eagle_nodes", 0)),
                    "sam_nodes": int(fused_tree.metadata.get("sam_nodes", 0)),
                    "final_nodes": int(fused_tree.metadata.get("final_nodes", 0)),
                })
            return Candidates(
                CandidateType.tree,
                tokens,
                candidate_tokens,
                fused_tree.buffers_kwargs,
            )

        # High confidence - skip SAM, use pure EAGLE
        metadata = {
            "mode": "rejection_boundary",
            "eagle_nodes": eagle_nonroot_nodes,
            "sam_nodes": 0,
            "final_nodes": eagle_nonroot_nodes,
            "selected_eagle": eagle_nonroot_nodes,
            "selected_sam": 0,
            "selected_both": 0,
            "node_sources": ["root"] + ["eagle"] * eagle_nonroot_nodes,
            "sam_skipped": True,
            "root_margin": float(root_margin),
            "rejection_conf_threshold": float(threshold),
        }
        draft.record_naive_fusion(metadata)
        if profiler is not None:
            profiler.add_step_metadata(metadata)
        return Candidates(
            type=CandidateType.tree,
            tokens=torch.tensor([eagle_tokens], dtype=torch.long, device=device),
            candidate_tokens=_extract_candidate_tokens_eagle(
                eagle_tokens,
                eagle_buffers,
                device,
            ),
            buffers_kwargs=eagle_buffers,
        )

    if samd_config.fusion_mode == "boundary_graft":
        from .fusion.boundary_graft import (
            predict_rejection_depth,
            extract_prefix_tokens_and_node,
            graft_sam_at_depth,
        )
        from .tree_model.fusion import TreeSpec

        timer = profiler.start_section("draft_eagle") if profiler is not None else 0.0
        try:
            eagle_tokens, eagle_buffers, eagle_logprobs = draft.tree_model.gen_draft(
                start_token,
                return_logprobs=True,
            )
        finally:
            if profiler is not None:
                profiler.end_section("draft_eagle", timer)

        # Predict rejection depth
        position_ids = eagle_buffers.get("tree_position_ids")
        predicted_depth = predict_rejection_depth(
            logprobs=eagle_logprobs,
            position_ids=position_ids,
            threshold=samd_config.boundary_graft_threshold,
            min_depth=samd_config.boundary_graft_min_depth,
            max_depth=samd_config.boundary_graft_max_depth,
        )

        eagle_nonroot_nodes = max(len(eagle_tokens) - 1, 0)

        if predicted_depth > 0:
            # Parse EAGLE tree
            eagle_tree_spec = TreeSpec.from_eagle3_buffers(
                eagle_tokens,
                eagle_buffers["tree_attn_mask"],
                eagle_buffers["tree_position_ids"],
            )

            # Extract EAGLE prefix and graft point
            prefix_tokens, parent_node_idx = extract_prefix_tokens_and_node(
                eagle_tree_spec, predicted_depth
            )

            # Validate prefix length
            if len(prefix_tokens) < predicted_depth:
                # Prefix too short, fall back to EAGLE-only
                metadata = {
                    "mode": "boundary_graft",
                    "predicted_depth": int(predicted_depth),
                    "threshold": float(samd_config.boundary_graft_threshold),
                    "eagle_nodes": eagle_nonroot_nodes,
                    "sam_nodes": 0,
                    "final_nodes": eagle_nonroot_nodes,
                    "sam_skipped": True,
                    "skip_reason": "prefix_too_short",
                }
                draft.record_naive_fusion(metadata)
                if profiler is not None:
                    profiler.add_step_metadata(metadata)
                return Candidates(
                    type=CandidateType.tree,
                    tokens=torch.tensor([eagle_tokens], dtype=torch.long, device=device),
                    candidate_tokens=_extract_candidate_tokens_eagle(
                        eagle_tokens,
                        eagle_buffers,
                        device,
                    ),
                    buffers_kwargs=eagle_buffers,
                )

            # SAM continuation from prefix
            timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
            try:
                if len(prefix_tokens) > 0:
                    # Transfer SAM state to prefix (excluding last token which is the graft point)
                    draft.sam_dyn.transfer_state(prefix_tokens[:-1])
                    draft.sam_static.transfer_state(prefix_tokens[:-1])
                    sam_start_token = prefix_tokens[-1]
                else:
                    sam_start_token = start_token

                index_dyn, match_dyn = draft.sam_dyn.lookup(sam_start_token)
                index_static, match_static_raw = draft.sam_static.lookup(sam_start_token)
                match_static = match_static_raw - draft.len_bias

                if match_dyn >= match_static:
                    sam_candidates = draft.sam_dyn.gen_draft_raw(
                        index_dyn,
                        sam_start_token,
                        samd_config.boundary_graft_max_sam_nodes,
                    )
                    sam_match_length = match_dyn
                else:
                    sam_candidates = draft.sam_static.gen_draft_raw(
                        index_static,
                        sam_start_token,
                        samd_config.boundary_graft_max_sam_nodes,
                    )
                    sam_match_length = max(match_static, 0)
            finally:
                if profiler is not None:
                    profiler.end_section("draft_sam", timer)

            # Graft SAM tail onto EAGLE tree
            timer = profiler.start_section("fusion_logic") if profiler is not None else 0.0
            try:
                fused_tree_spec = graft_sam_at_depth(
                    eagle_tree=eagle_tree_spec,
                    sam_candidates=sam_candidates,
                    parent_node_idx=parent_node_idx,
                    max_added_nodes=samd_config.boundary_graft_max_sam_nodes,
                )

                # Convert back to buffers
                tree_attn_mask, tree_position_ids, tree_retrieve_indices = (
                    fused_tree_spec.to_buffer_lists()
                )

                fused_tokens = torch.tensor(
                    fused_tree_spec.tokens,
                    dtype=torch.long,
                    device=device,
                )
                tokens = fused_tokens.unsqueeze(0)

                tokens_ext = torch.cat([fused_tokens, torch.zeros(1, dtype=torch.long, device=device)])
                retrieve_indices_tensor = torch.tensor(
                    tree_retrieve_indices,
                    dtype=torch.long,
                    device=device,
                )
                candidate_tokens = tokens_ext[retrieve_indices_tensor]

                fused_buffers = {
                    "tree_attn_mask": torch.tensor(tree_attn_mask, dtype=torch.bool, device=device),
                    "tree_position_ids": torch.tensor(tree_position_ids, dtype=torch.long, device=device),
                    "tree_retrieve_indices": retrieve_indices_tensor,
                }

                sam_nodes_added = len(fused_tree_spec.tokens) - len(eagle_tree_spec.tokens)
                metadata = {
                    "mode": "boundary_graft",
                    "predicted_depth": int(predicted_depth),
                    "threshold": float(samd_config.boundary_graft_threshold),
                    "eagle_nodes": eagle_nonroot_nodes,
                    "sam_nodes": sam_nodes_added,
                    "final_nodes": len(fused_tree_spec.tokens) - 1,
                    "sam_match_length": sam_match_length,
                    "prefix_length": len(prefix_tokens),
                    "sam_skipped": False,
                }
            finally:
                if profiler is not None:
                    profiler.end_section("fusion_logic", timer)

            draft.record_naive_fusion(metadata)
            if profiler is not None:
                profiler.add_step_metadata(metadata)

            return Candidates(
                CandidateType.tree,
                tokens,
                candidate_tokens,
                fused_buffers,
            )

        # No prediction - use pure EAGLE
        metadata = {
            "mode": "boundary_graft",
            "predicted_depth": -1,
            "threshold": float(samd_config.boundary_graft_threshold),
            "eagle_nodes": eagle_nonroot_nodes,
            "sam_nodes": 0,
            "final_nodes": eagle_nonroot_nodes,
            "sam_skipped": True,
            "skip_reason": "no_prediction",
        }
        draft.record_naive_fusion(metadata)
        if profiler is not None:
            profiler.add_step_metadata(metadata)

        return Candidates(
            type=CandidateType.tree,
            tokens=torch.tensor([eagle_tokens], dtype=torch.long, device=device),
            candidate_tokens=_extract_candidate_tokens_eagle(
                eagle_tokens,
                eagle_buffers,
                device,
            ),
            buffers_kwargs=eagle_buffers,
        )

    if samd_config.fusion_mode != "none":
        raise ValueError("unsupported fusion_mode: {}".format(samd_config.fusion_mode))

    candidate_type, tokens, buffers_kwargs = draft.lookup(start_token)
    tree_retrieve_indices = buffers_kwargs.get("tree_retrieve_indices", tree_retrieve_indices)
    if candidate_type == CandidateType.sequence:
        tokens = torch.tensor([tokens], dtype=torch.long, device=device)
        candidate_tokens = tokens
    else:
        tokens_ext = torch.tensor(tokens + [0], dtype=torch.long, device=device)
        candidate_tokens = tokens_ext[tree_retrieve_indices]
        tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    return Candidates(
        candidate_type,
        tokens,
        candidate_tokens,
        buffers_kwargs,
    )


@profile_decorator("eval_posterior")
def eval_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    config: SamdGenerationConfig,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    if config.greedy:
        # Greedy decoding based on temperature value
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length + 1, logits[best_candidate, accept_length].view(1, -1)
    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = config.logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        sample_p = sample_p.view(1, -1)
        accept_length = torch.tensor(accept_length, dtype=torch.long, device=candidates.device)
        best_candidate = torch.tensor(best_candidate, dtype=torch.long, device=candidates.device)
        return best_candidate, accept_length, sample_p
