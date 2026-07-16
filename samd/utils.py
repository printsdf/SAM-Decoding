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
from .fusion.naive_fusion import fuse_eagle_sam_naive


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

    if samd_config.fusion_mode == "drafter_mars":
        from .fusion.drafter_mars_gate import top_path_ratio_trigger
        from .tree_model.fusion import TreeSpec, eagle3_parents_from_buffers

        # Adaptive-theta controller (per request, created in DraftModel.reset);
        # None unless drafter_mars_adaptive_theta is on — metadata's
        # drafter_mars_theta always records the theta actually used.
        controller = getattr(draft, "drafter_mars_controller", None)
        theta = (
            controller.theta
            if controller is not None
            else samd_config.drafter_mars_theta
        )
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
                "repair": samd_config.drafter_mars_repair,
                "ratio_triggered": False,
                "max_top_path_ratio": None,
                "trigger_depth": None,
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
            eagle_parents = eagle3_parents_from_buffers(
                eagle_buffers["tree_attn_mask"],
                eagle_buffers["tree_position_ids"],
            )
            ratio_triggered, max_top_path_ratio = top_path_ratio_trigger(
                eagle_parents,
                eagle_logprobs,
                eagle_raw_logits,
                theta,
            )
        finally:
            if profiler is not None:
                profiler.end_section("fusion_logic", timer)

        # Only steps that reach the ratio gate update the trigger-rate EMA.
        if controller is not None:
            controller.update(ratio_triggered)

        sam_candidates = None
        sam_match_length = 0
        repair = samd_config.drafter_mars_repair
        trigger_depth = None
        skip_reason = None

        if (
            ratio_triggered
            and repair == "graft"
            and samd_config.drafter_mars_max_grafts == 1
        ):
            from .fusion.drafter_mars_gate import earliest_top_path_trigger

            trigger = earliest_top_path_trigger(
                eagle_parents,
                eagle_logprobs,
                eagle_raw_logits,
                theta,
            )
            if trigger is None:
                skip_reason = "no_trigger_info"
            else:
                trigger_depth = int(trigger.parent_depth)
                # Greedy top-path prefix: root .. triggering parent (inclusive).
                chain = []
                node = trigger.parent_index
                while node != -1:
                    chain.append(node)
                    node = eagle_parents[node]
                chain.reverse()
                prefix_tokens = [int(eagle_tokens[i]) for i in chain]
                sam_start_token = prefix_tokens[-1]

                timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
                try:
                    # Stateless prefix walk: advance functional SAM states from
                    # the current context state along the drafted prefix. The
                    # SAMs' cur state is never mutated, so nothing to restore;
                    # the post-verification draft.update re-syncs as usual.
                    dyn_index, dyn_length = (
                        draft.sam_dyn.cur_index,
                        draft.sam_dyn.cur_length,
                    )
                    static_index, static_length = (
                        draft.sam_static.cur_index,
                        draft.sam_static.cur_length,
                    )
                    for token in prefix_tokens:
                        dyn_index, dyn_length = draft.sam_dyn.transfer_state(
                            dyn_index, dyn_length, token
                        )
                        static_index, static_length = draft.sam_static.transfer_state(
                            static_index, static_length, token
                        )
                    match_static_prefix = static_length - draft.len_bias
                    if dyn_length >= match_static_prefix:
                        sam_candidates = draft.sam_dyn.gen_draft_raw(
                            dyn_index,
                            sam_start_token,
                            samd_config.n_predicts,
                        )
                        sam_match_length = int(dyn_length)
                    else:
                        sam_candidates = draft.sam_static.gen_draft_raw(
                            static_index,
                            sam_start_token,
                            samd_config.n_predicts,
                        )
                        sam_match_length = max(int(match_static_prefix), 0)
                finally:
                    if profiler is not None:
                        profiler.end_section("draft_sam", timer)

                if sam_candidates is None or len(sam_candidates) <= 1:
                    skip_reason = "empty_sam_continuation"
                else:
                    timer = (
                        profiler.start_section("fusion_logic")
                        if profiler is not None
                        else 0.0
                    )
                    try:
                        # sam_candidates[0] is sam_start_token (the triggering
                        # parent's token, already in the tree); graft only the
                        # continuation at that parent, author walk-reuse
                        # semantics, full n_predicts horizon.
                        tree_spec = TreeSpec(
                            tokens=[int(token) for token in eagle_tokens],
                            parents=list(eagle_parents),
                        )
                        fused_tree_spec = tree_spec.graft_sequence_at(
                            trigger.parent_index,
                            sam_candidates[1:],
                        )
                        sam_nodes_added = len(fused_tree_spec.tokens) - len(
                            tree_spec.tokens
                        )
                        if sam_nodes_added > 0:
                            fused_buffers = fused_tree_spec.to_buffers(
                                device=device,
                                mask_dtype=eagle_buffers["tree_attn_mask"].dtype,
                            )
                            fused_tokens = torch.tensor(
                                fused_tree_spec.tokens,
                                dtype=torch.long,
                                device=device,
                            )
                            tokens = fused_tokens.unsqueeze(0)
                            tokens_ext = torch.cat(
                                [
                                    fused_tokens,
                                    torch.zeros(1, dtype=torch.long, device=device),
                                ]
                            )
                            candidate_tokens = tokens_ext[
                                fused_buffers["tree_retrieve_indices"]
                            ]
                    finally:
                        if profiler is not None:
                            profiler.end_section("fusion_logic", timer)

                    if sam_nodes_added > 0:
                        metadata = {
                            "mode": "drafter_mars",
                            "repair": "graft",
                            "ratio_triggered": True,
                            "max_top_path_ratio": float(max_top_path_ratio),
                            "trigger_depth": trigger_depth,
                            "trigger_ratio": float(trigger.ratio),
                            "drafter_mars_theta": float(theta),
                            "eagle_nodes": eagle_nonroot_nodes,
                            "sam_nodes": int(sam_nodes_added),
                            "final_nodes": len(fused_tree_spec.tokens) - 1,
                            "sam_match_length": int(sam_match_length),
                            "prefix_length": len(prefix_tokens),
                            "sam_skipped": False,
                        }
                        draft.record_naive_fusion(metadata)
                        if profiler is not None:
                            profiler.add_step_metadata(metadata)
                        return Candidates(
                            CandidateType.tree,
                            tokens,
                            candidate_tokens,
                            fused_buffers,
                        )
                    skip_reason = "graft_added_nothing"

        if (
            ratio_triggered
            and repair == "graft"
            and samd_config.drafter_mars_max_grafts > 1
        ):
            from .fusion.drafter_mars_gate import all_top_path_triggers

            triggers = all_top_path_triggers(
                eagle_parents,
                eagle_logprobs,
                eagle_raw_logits,
                theta,
            )
            selected = []
            timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
            try:
                for trigger in triggers:
                    if len(selected) >= samd_config.drafter_mars_max_grafts:
                        break
                    chain = []
                    node = trigger.parent_index
                    while node != -1:
                        chain.append(node)
                        node = eagle_parents[node]
                    chain.reverse()
                    prefix_tokens = [int(eagle_tokens[i]) for i in chain]
                    dyn_index, dyn_length = (
                        draft.sam_dyn.cur_index,
                        draft.sam_dyn.cur_length,
                    )
                    static_index, static_length = (
                        draft.sam_static.cur_index,
                        draft.sam_static.cur_length,
                    )
                    for token in prefix_tokens:
                        dyn_index, dyn_length = draft.sam_dyn.transfer_state(
                            dyn_index, dyn_length, token
                        )
                        static_index, static_length = draft.sam_static.transfer_state(
                            static_index, static_length, token
                        )
                    match_static_prefix = static_length - draft.len_bias
                    if dyn_length >= match_static_prefix:
                        continuation = draft.sam_dyn.gen_draft_raw(
                            dyn_index, prefix_tokens[-1], samd_config.n_predicts
                        )
                        match_length = int(dyn_length)
                    else:
                        continuation = draft.sam_static.gen_draft_raw(
                            static_index, prefix_tokens[-1], samd_config.n_predicts
                        )
                        match_length = max(int(match_static_prefix), 0)
                    if continuation is None or len(continuation) <= 1:
                        # Empty continuation does not consume a graft slot.
                        continue
                    selected.append(
                        (trigger, continuation[1:], match_length, len(prefix_tokens))
                    )
            finally:
                if profiler is not None:
                    profiler.end_section("draft_sam", timer)

            if not selected:
                skip_reason = "empty_sam_continuation"
            else:
                timer = (
                    profiler.start_section("fusion_logic")
                    if profiler is not None
                    else 0.0
                )
                try:
                    fused_tree_spec = TreeSpec(
                        tokens=[int(token) for token in eagle_tokens],
                        parents=list(eagle_parents),
                    )
                    base_nodes = len(fused_tree_spec.tokens)
                    grafts_meta = []
                    # Author walk-reuse semantics; grafts only append nodes,
                    # so original parent indices stay valid across sequential
                    # grafts.
                    for trigger, continuation, _match, _prefix in selected:
                        before = len(fused_tree_spec.tokens)
                        fused_tree_spec = fused_tree_spec.graft_sequence_at(
                            trigger.parent_index,
                            continuation,
                        )
                        grafts_meta.append({
                            "depth": int(trigger.parent_depth),
                            "ratio": float(trigger.ratio),
                            "nodes_added": len(fused_tree_spec.tokens) - before,
                        })
                    sam_nodes_added = len(fused_tree_spec.tokens) - base_nodes
                    if sam_nodes_added > 0:
                        fused_buffers = fused_tree_spec.to_buffers(
                            device=device,
                            mask_dtype=eagle_buffers["tree_attn_mask"].dtype,
                        )
                        fused_tokens = torch.tensor(
                            fused_tree_spec.tokens,
                            dtype=torch.long,
                            device=device,
                        )
                        tokens = fused_tokens.unsqueeze(0)
                        tokens_ext = torch.cat(
                            [
                                fused_tokens,
                                torch.zeros(1, dtype=torch.long, device=device),
                            ]
                        )
                        candidate_tokens = tokens_ext[
                            fused_buffers["tree_retrieve_indices"]
                        ]
                finally:
                    if profiler is not None:
                        profiler.end_section("fusion_logic", timer)

                if sam_nodes_added > 0:
                    metadata = {
                        "mode": "drafter_mars",
                        "repair": "graft",
                        "ratio_triggered": True,
                        "max_top_path_ratio": float(max_top_path_ratio),
                        "trigger_depth": int(selected[0][0].parent_depth),
                        "trigger_ratio": float(selected[0][0].ratio),
                        "drafter_mars_theta": float(theta),
                        "graft_count": len(grafts_meta),
                        "grafts": grafts_meta,
                        "eagle_nodes": eagle_nonroot_nodes,
                        "sam_nodes": int(sam_nodes_added),
                        "final_nodes": len(fused_tree_spec.tokens) - 1,
                        "sam_match_length": int(max(item[2] for item in selected)),
                        "prefix_length": int(selected[0][3]),
                        "sam_skipped": False,
                    }
                    draft.record_naive_fusion(metadata)
                    if profiler is not None:
                        profiler.add_step_metadata(metadata)
                    return Candidates(
                        CandidateType.tree,
                        tokens,
                        candidate_tokens,
                        fused_buffers,
                    )
                skip_reason = "graft_added_nothing"

        if ratio_triggered and repair == "naive_fuse":
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

        if (
            ratio_triggered
            and repair == "naive_fuse"
            and sam_candidates is not None
            and len(sam_candidates) > 1
        ):
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
                fused_tree.metadata["repair"] = "naive_fuse"
                fused_tree.metadata["ratio_triggered"] = True
                fused_tree.metadata["max_top_path_ratio"] = float(max_top_path_ratio)
                fused_tree.metadata["trigger_depth"] = trigger_depth
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
                    "repair": "naive_fuse",
                    "ratio_triggered": True,
                    "max_top_path_ratio": float(max_top_path_ratio),
                    "trigger_depth": trigger_depth,
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

        # Leaf extension (drafter confident / repair unavailable): graft the
        # SAM continuation at the greedy leaf, same n_predicts horizon as the
        # baseline sequence draft — the tree keeps its paths, the greedy path
        # gains up to n_predicts extra depth.
        if samd_config.drafter_mars_extend:
            from .fusion.drafter_mars_gate import greedy_top_path

            path = greedy_top_path(eagle_parents, eagle_logprobs)
            leaf = path[-1] if path else 0
            if leaf != 0:
                prefix_tokens = [int(eagle_tokens[i]) for i in path]
                timer = profiler.start_section("draft_sam") if profiler is not None else 0.0
                try:
                    dyn_index, dyn_length = (
                        draft.sam_dyn.cur_index,
                        draft.sam_dyn.cur_length,
                    )
                    static_index, static_length = (
                        draft.sam_static.cur_index,
                        draft.sam_static.cur_length,
                    )
                    for token in prefix_tokens:
                        dyn_index, dyn_length = draft.sam_dyn.transfer_state(
                            dyn_index, dyn_length, token
                        )
                        static_index, static_length = draft.sam_static.transfer_state(
                            static_index, static_length, token
                        )
                    match_static_prefix = static_length - draft.len_bias
                    if dyn_length >= match_static_prefix:
                        extension = draft.sam_dyn.gen_draft_raw(
                            dyn_index, prefix_tokens[-1], samd_config.n_predicts
                        )
                        extend_match = int(dyn_length)
                    else:
                        extension = draft.sam_static.gen_draft_raw(
                            static_index, prefix_tokens[-1], samd_config.n_predicts
                        )
                        extend_match = max(int(match_static_prefix), 0)
                finally:
                    if profiler is not None:
                        profiler.end_section("draft_sam", timer)

                if extension is not None and len(extension) > 1:
                    timer = (
                        profiler.start_section("fusion_logic")
                        if profiler is not None
                        else 0.0
                    )
                    try:
                        tree_spec = TreeSpec(
                            tokens=[int(token) for token in eagle_tokens],
                            parents=list(eagle_parents),
                        )
                        fused_tree_spec = tree_spec.graft_sequence_at(
                            leaf,
                            extension[1:],
                        )
                        extend_nodes = len(fused_tree_spec.tokens) - len(
                            tree_spec.tokens
                        )
                        if extend_nodes > 0:
                            fused_buffers = fused_tree_spec.to_buffers(
                                device=device,
                                mask_dtype=eagle_buffers["tree_attn_mask"].dtype,
                            )
                            fused_tokens = torch.tensor(
                                fused_tree_spec.tokens,
                                dtype=torch.long,
                                device=device,
                            )
                            tokens = fused_tokens.unsqueeze(0)
                            tokens_ext = torch.cat(
                                [
                                    fused_tokens,
                                    torch.zeros(1, dtype=torch.long, device=device),
                                ]
                            )
                            candidate_tokens = tokens_ext[
                                fused_buffers["tree_retrieve_indices"]
                            ]
                    finally:
                        if profiler is not None:
                            profiler.end_section("fusion_logic", timer)

                    if extend_nodes > 0:
                        metadata = {
                            "mode": "drafter_mars",
                            "repair": repair,
                            "ratio_triggered": bool(ratio_triggered),
                            "max_top_path_ratio": (
                                float(max_top_path_ratio)
                                if max_top_path_ratio is not None
                                else None
                            ),
                            "trigger_depth": trigger_depth,
                            "drafter_mars_theta": float(theta),
                            "extended": True,
                            "extend_depth": len(prefix_tokens) - 1,
                            "eagle_nodes": eagle_nonroot_nodes,
                            "sam_nodes": int(extend_nodes),
                            "final_nodes": len(fused_tree_spec.tokens) - 1,
                            "sam_match_length": int(extend_match),
                            "prefix_length": len(prefix_tokens),
                            "sam_skipped": False,
                        }
                        if skip_reason is not None:
                            metadata["skip_reason"] = skip_reason
                        draft.record_naive_fusion(metadata)
                        if profiler is not None:
                            profiler.add_step_metadata(metadata)
                        return Candidates(
                            CandidateType.tree,
                            tokens,
                            candidate_tokens,
                            fused_buffers,
                        )

        # Not triggered (or SAM repair unavailable): pure EAGLE3 tree.
        metadata = {
            "mode": "drafter_mars",
            "repair": repair,
            "ratio_triggered": bool(ratio_triggered),
            "max_top_path_ratio": (
                float(max_top_path_ratio) if max_top_path_ratio is not None else None
            ),
            "trigger_depth": trigger_depth,
            "drafter_mars_theta": float(theta),
            "extended": False,
            "eagle_nodes": eagle_nonroot_nodes,
            "sam_nodes": 0,
            "final_nodes": eagle_nonroot_nodes,
            "selected_eagle": eagle_nonroot_nodes,
            "selected_sam": 0,
            "selected_both": 0,
            "node_sources": ["root"] + ["eagle"] * eagle_nonroot_nodes,
            "sam_skipped": True,
        }
        if skip_reason is not None:
            metadata["skip_reason"] = skip_reason
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
