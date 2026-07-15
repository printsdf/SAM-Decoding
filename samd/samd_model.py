import os
import json
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Optional, Union, List, Literal, Tuple, Dict, Any
from types import MethodType
from transformers import LlamaForCausalLM, LlamaTokenizer
from .samd_config import SamdConfig, ForwardState, ForwardType, MaskState
from .utils import (
    OptionalTensor,
    CandidateType,
    SamdGenerationConfig,
    gen_candidates,
    eval_posterior,
)
from .cache import SamdCache, SamdStaticCache
from .draft import DraftModel
from .diagnosis import make_trace_step
from .model_patch import patch_dict, attn_patch_dict, eagle3_patch_dict, eagle3_attn_patch_dict
from profile_utils import profile_decorator, profile_accept_length

Outputs = namedtuple(
    'Outputs',
    ['output_ids', 'decode_tokens', 'decode_steps',
     'accepet_length_per_step', 'diagnosis_trace'],
    defaults=[None],
)

class SamdModel(nn.Module):
    
    def __init__(self,
        samd_config: SamdConfig,
        lm: LlamaForCausalLM,
        draft: DraftModel,
        eos_token_id: int,
        dtype: torch.dtype,
        device: str,
        stop_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.samd_config = samd_config
        self.gen_config: SamdGenerationConfig = None
        self.eos_token = eos_token_id
        self.stop_token = stop_token_id

        self.lm = lm
        self.draft = draft
        self.dtype = dtype
        self.device = device
        
        # buffers
        self.seq_position_ids: torch.Tensor = None
        self.base_tree_attn_mask: torch.Tensor = None
        self.base_tree_position_ids: torch.Tensor = None
        self.base_tree_retrieve_indices: torch.Tensor = None
        self.tree_attn_mask: torch.Tensor = None
        self.tree_position_ids: torch.Tensor = None
        self.tree_retrieve_indices: torch.Tensor = None
        
        # buffers
        self.cache: Union[SamdCache, SamdStaticCache] = None
        self.forward_state = ForwardState(None)
        self.mask_state = MaskState(None)
        
        self.init_buffers()
        self.register_forward_patch()

    def register_forward_patch(self):
        if self.samd_config.tree_method == "eagle3":
            _patch_dict = eagle3_patch_dict
            _attn_patch_dict = eagle3_attn_patch_dict
        else:
            _patch_dict = patch_dict
            _attn_patch_dict = attn_patch_dict
        for module_name, module in self.lm.named_modules():
            module_name = "root" if module_name == "" else "root.{}".format(module_name)
            if type(module) in _patch_dict:
                for fn_name, fn in _patch_dict[type(module)]:
                    setattr(module, fn_name, MethodType(fn, module))
                    print("setattr {} -> {}".format(module_name, fn_name))
            if type(module) in _attn_patch_dict:
                for fn_name, fn in _attn_patch_dict[type(module)]:
                    setattr(module, fn_name, MethodType(fn, module))
                    setattr(module, "mask_state", self.mask_state)
                    setattr(module, "forward_state", self.forward_state)
                    print("attn setattr {} -> {}".format(module_name, fn_name))

    
    def init_seq_position_ids(self):
        return torch.tensor(
            range(0, self.samd_config.n_predicts), 
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)
    
    def init_buffers(self):
        self.seq_position_ids = self.init_seq_position_ids()
        buffers = self.draft.tree_model.gen_buffers()
        self.base_tree_attn_mask = buffers["tree_attn_mask"]
        self.base_tree_position_ids = buffers["tree_position_ids"]
        self.base_tree_retrieve_indices = buffers["tree_retrieve_indices"]
        self.mask_state.set_state(self.base_tree_attn_mask)
    
    def update_buffers(self, buffers_kwargs: Dict[str, Optional[torch.Tensor]]):
        self.tree_attn_mask = buffers_kwargs.get("tree_attn_mask", self.base_tree_attn_mask)
        self.tree_position_ids = buffers_kwargs.get("tree_position_ids", self.base_tree_position_ids)
        self.tree_retrieve_indices = buffers_kwargs.get("tree_retrieve_indices", self.base_tree_retrieve_indices)
        self.mask_state.set_state(self.tree_attn_mask)
    
    # @profile_decorator("SamdModel.prefill")
    def prefill(self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
    ):
        self.forward_state.forward_type = ForwardType.prefill
        outputs = self.lm(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            past_key_values=self.cache,
        )
        logits = outputs.logits
        last_hidden_states = outputs.last_hidden_states \
            if self.samd_config.use_last_hidden_states else None
        last_hidden_states = OptionalTensor(last_hidden_states).apply(
            lambda x: x.squeeze(0)
        ).data
        self.draft.update(
            tokens=input_ids.squeeze(0),
            last_hidden_states=last_hidden_states,
            tree_tokens=input_ids.squeeze(0),
            tree_logits=logits.squeeze(0)
        )
        self.cache.set_length()
        if self.gen_config.greedy:
            sample_p = logits[:, -1]
        else:
            sample_p = torch.softmax(logits[:, -1], dim=-1)
        return sample_p  # [1, D]
    
    # @profile_decorator("SamdModel.decode")
    def decode(self, sample_p: torch.Tensor, length: int):
        profiler = getattr(self.gen_config, "fusion_profiler", None)
        if profiler is not None and not getattr(profiler, "enabled", False):
            profiler = None
        if profiler is not None:
            profiler.start_step()
        try:
            candidates = gen_candidates(
                sample_p,
                self.base_tree_retrieve_indices,
                self.draft,
                self.samd_config,
                self.gen_config,
                self.device
            )
            if profiler is not None:
                profiler.add_step_metadata({
                    "candidate_type": getattr(candidates.type, "value", str(candidates.type))
                })

            timer = profiler.start_section("verify") if profiler is not None else 0.0
            try:
                self.update_buffers(candidates.buffers_kwargs)
                if candidates.type == CandidateType.sequence:
                    self.forward_state.forward_type = ForwardType.seq_decode
                    position_ids = self.seq_position_ids + length
                else:
                    self.forward_state.forward_type = ForwardType.tree_decode
                    position_ids = self.tree_position_ids + length
                input_ids = candidates.tokens
                outputs = self.lm(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    past_key_values=self.cache,
                )
                tree_logits = outputs.logits
                # print("tree_logits.shape:", tree_logits.shape)
                if self.samd_config.use_last_hidden_states:
                    tree_last_hidden_states = OptionalTensor(outputs.last_hidden_states)
                else:
                    tree_last_hidden_states = OptionalTensor(None)
                if candidates.type == CandidateType.sequence:
                    candidate_logits = tree_logits
                    candidate_last_hidden_states = tree_last_hidden_states
                    candidate_indices = OptionalTensor(None)
                else:
                    candidate_logits = tree_logits.squeeze(0)[self.tree_retrieve_indices]
                    candidate_last_hidden_states = tree_last_hidden_states.apply(
                        lambda x: x.squeeze(0)[self.tree_retrieve_indices]
                    )
                    candidate_indices = OptionalTensor(self.tree_retrieve_indices)

                best_candidate, accept_length, sample_p \
                    = eval_posterior(candidate_logits, candidates.candidate_tokens, self.gen_config)
                fusion_meta = None
                if self.samd_config.fusion_mode == "naive":
                    fusion_meta = self.draft.fusion_stats.get("last_step")
                accepted_indices = candidate_indices.apply(
                    lambda x: x[best_candidate][:accept_length]
                ).data
                new_tokens = self.update_state(
                    input_ids.squeeze(0),
                    tree_logits.squeeze(0),
                    best_candidate,
                    accept_length,
                    candidates.candidate_tokens,
                    candidate_indices,
                    candidate_last_hidden_states,
                    fusion_meta=fusion_meta,
                    accepted_indices=accepted_indices,
                )
                if profiler is not None and fusion_meta is not None:
                    oracle_trace = self._build_oracle_profile_trace(
                        fusion_meta,
                        new_tokens,
                        accepted_indices,
                    )
                    if oracle_trace is not None:
                        profiler.add_step_trace(**oracle_trace)
            finally:
                if profiler is not None:
                    profiler.end_section("verify", timer)
            # print("new_tokens:\n{}".format(new_tokens))
            if self.gen_config.collect_diagnosis_trace:
                accept_length_int = int(
                    accept_length.detach().cpu().item()
                    if hasattr(accept_length, "item") else accept_length
                )
                if candidates.type == CandidateType.tree:
                    trace_meta: Optional[Dict[str, Any]] = {
                        "path_type": "tree",
                        "best_candidate": int(best_candidate.detach().cpu().item()),
                        "accept_length": accept_length_int,
                        "candidates": candidates.candidate_tokens,
                        "tree_logits": candidate_logits,
                    }
                else:
                    trace_meta = {
                        "path_type": "sequence",
                        "best_candidate": None,
                        "accept_length": accept_length_int,
                        "candidates": None,
                        "tree_logits": None,
                    }
            else:
                trace_meta = None
            if profiler is not None:
                profiler.finish_step({
                    "accepted_tokens": len(new_tokens),
                    "new_tokens": list(new_tokens),
                })
            return sample_p, new_tokens, trace_meta
        except Exception as exc:
            if profiler is not None:
                profiler.abort_step(str(exc))
            raise

    # @profile_decorator("SamdModel.update_state")
    def update_state(self,
        tree_tokens: torch.Tensor,
        tree_logits: torch.Tensor,
        best_candidate: torch.Tensor, 
        accept_length: torch.Tensor,
        candiate_tokens: torch.Tensor,
        candidate_indices: OptionalTensor,
        candidate_last_hidden_states: OptionalTensor,
        fusion_meta: Optional[Dict[str, Any]] = None,
        accepted_indices: Optional[torch.Tensor] = None,
    ):
        tokens = candiate_tokens[best_candidate][:accept_length]
        
        indices: Optional[torch.Tensor] = accepted_indices
        if indices is None:
            indices = candidate_indices.apply(
                lambda x: x[best_candidate][:accept_length]
            ).data
        last_hidden_states: Optional[torch.Tensor] = candidate_last_hidden_states.apply(
            lambda x: x[best_candidate][:accept_length]
        ).data
        
        self.draft.update(
            tokens=tokens, 
            last_hidden_states=last_hidden_states,
            tree_tokens=tree_tokens,
            tree_logits=tree_logits,
            fusion_meta=fusion_meta,
            accepted_indices=indices,
        )
        self.cache.select_indices(indices, accept_length.item())
        
        return tokens.tolist()

    @staticmethod
    def _int_list(value: Any) -> List[int]:
        if value is None:
            return []
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, list):
            value = [value]
        flattened: List[int] = []
        for item in value:
            if isinstance(item, list):
                flattened.extend(SamdModel._int_list(item))
            else:
                flattened.append(int(item))
        return flattened

    @staticmethod
    def _candidate_nonroot_path(candidate: Dict[str, Any]) -> List[int]:
        token_path = SamdModel._int_list(candidate.get("token_path"))
        if token_path:
            depth = int(candidate.get("depth", max(len(token_path) - 1, 0)))
            if len(token_path) >= depth + 1:
                return [int(token) for token in token_path[-depth:]]
            return [int(token) for token in token_path]
        path = SamdModel._int_list(candidate.get("path"))
        if path:
            parent_nonroot = path[1:]
            depth = int(candidate.get("depth", len(parent_nonroot) + 1))
            parent_depth = max(depth - 1, 0)
            if parent_depth == 0:
                parent_nonroot = []
            elif len(parent_nonroot) > parent_depth:
                parent_nonroot = parent_nonroot[-parent_depth:]
            token = candidate.get("token")
            if token is None:
                return parent_nonroot
            return parent_nonroot + [int(token)]
        token = candidate.get("token")
        return [int(token)] if token is not None else []

    @classmethod
    def _build_rejection_boundary_labels(
        cls,
        candidates: List[Dict[str, Any]],
        accepted_path: List[int],
    ) -> Dict[str, Any]:
        eagle_candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("source")) == "eagle"
        ]
        by_nonroot_path: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        children_by_parent_prefix: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
        for candidate in eagle_candidates:
            nonroot_path = tuple(cls._candidate_nonroot_path(candidate))
            if not nonroot_path:
                continue
            by_nonroot_path.setdefault(nonroot_path, candidate)
            children_by_parent_prefix.setdefault(nonroot_path[:-1], []).append(candidate)

        if not accepted_path and eagle_candidates:
            return {
                "first_rejected_depth": 1,
                "first_rejected_tree_index": None,
                "first_rejected_parent_index": 0,
                "first_rejected_parent_path": [],
                "first_rejected_candidate_count": len(children_by_parent_prefix.get((), [])),
            }

        for depth in range(1, len(accepted_path) + 1):
            prefix = tuple(int(token) for token in accepted_path[:depth])
            if prefix in by_nonroot_path:
                continue
            parent_prefix = prefix[:-1]
            parent_candidate = by_nonroot_path.get(parent_prefix)
            if depth == 1:
                parent_index = 0
            elif parent_candidate is not None and parent_candidate.get("tree_index") is not None:
                parent_index = int(parent_candidate["tree_index"])
            else:
                parent_index = None

            # There is no concrete rejected tree node when the target token was
            # never proposed under the accepted parent; label the parent instead.
            rejected_tree_index = None
            if prefix in by_nonroot_path and by_nonroot_path[prefix].get("tree_index") is not None:
                rejected_tree_index = int(by_nonroot_path[prefix]["tree_index"])

            return {
                "first_rejected_depth": int(depth),
                "first_rejected_tree_index": rejected_tree_index,
                "first_rejected_parent_index": parent_index,
                "first_rejected_parent_path": [int(token) for token in parent_prefix],
                "first_rejected_candidate_count": len(
                    children_by_parent_prefix.get(parent_prefix, [])
                ),
            }

        return {
            "first_rejected_depth": None,
            "first_rejected_tree_index": None,
            "first_rejected_parent_index": None,
            "first_rejected_parent_path": None,
            "first_rejected_candidate_count": 0,
        }

    def _build_oracle_profile_trace(
        self,
        fusion_meta: Dict[str, Any],
        new_tokens: List[int],
        accepted_indices: Optional[torch.Tensor],
    ) -> Optional[Dict[str, Any]]:
        raw_candidates = fusion_meta.get("oracle_candidates")
        if not raw_candidates:
            return None

        accepted_path = [int(token) for token in new_tokens[1:]]
        accepted_index_set = set(self._int_list(accepted_indices))
        candidates = []
        for raw_candidate in raw_candidates:
            candidate = dict(raw_candidate)
            token_path = self._int_list(candidate.get("token_path"))
            nonroot_path = token_path[1:] if len(token_path) > 1 else []
            path_accepted = (
                len(nonroot_path) > 0
                and len(nonroot_path) <= len(accepted_path)
                and nonroot_path == accepted_path[: len(nonroot_path)]
            )
            tree_index = candidate.get("tree_index")
            index_accepted = (
                tree_index is not None and int(tree_index) in accepted_index_set
            )
            candidate["accepted"] = bool(path_accepted or index_accepted)
            candidates.append(candidate)
        labels = self._build_rejection_boundary_labels(candidates, accepted_path)

        return {
            "candidates": candidates,
            "acceptance_path": accepted_path,
            "labels": labels,
            "stats": {
                "mat": len(new_tokens),
                "accepted_nonroot": len(accepted_path),
                "eagle_nodes": int(fusion_meta.get("eagle_nodes", 0)),
                "sam_nodes": int(fusion_meta.get("sam_nodes", 0)),
                "final_nodes": int(fusion_meta.get("final_nodes", 0)),
                "sam_skipped": bool(fusion_meta.get("sam_skipped", False)),
            },
        }

    def _print_fusion_analysis(self) -> None:
        stats = self.draft.fusion_stats
        if not stats or stats.get("mode") != "naive" or stats.get("steps", 0) <= 0:
            return

        steps = max(int(stats.get("steps", 0)), 1)
        accept_rates = stats.get("accept_rates", [])
        total_eagle_accepted = sum(int(rate.get("eagle_accepted", 0)) for rate in accept_rates)
        total_sam_accepted = sum(int(rate.get("sam_accepted", 0)) for rate in accept_rates)
        total_accepted = total_eagle_accepted + total_sam_accepted
        avg_eagle_rate = (
            sum(float(rate.get("eagle_rate", 0.0)) for rate in accept_rates) / len(accept_rates)
            if accept_rates else 0.0
        )
        avg_sam_rate = (
            sum(float(rate.get("sam_rate", 0.0)) for rate in accept_rates) / len(accept_rates)
            if accept_rates else 0.0
        )
        last = stats.get("last_step") or {}

        print("\n" + "=" * 60)
        print("Naive Fusion Analysis")
        print("=" * 60)
        print("Average nodes per step:")
        print("  Eagle: {:.1f}".format(stats.get("eagle_nodes_sum", 0) / steps))
        print("  SAM:   {:.1f}".format(stats.get("sam_nodes_sum", 0) / steps))
        print("  Merged: {:.1f}".format(stats.get("merged_nodes_before_truncation_sum", 0) / steps))
        print("  Final:  {:.1f}".format(stats.get("final_nodes_sum", 0) / steps))
        print("")
        print("Average score/depth:")
        print(
            "  Eagle score/depth: {:.3f} / {:.2f}".format(
                float(stats.get("eagle_score_sum", 0.0)) / steps,
                float(stats.get("eagle_depth_sum", 0.0)) / steps,
            )
        )
        print(
            "  SAM score/depth:   {:.3f} / {:.2f}".format(
                float(stats.get("sam_score_sum", 0.0)) / steps,
                float(stats.get("sam_depth_sum", 0.0)) / steps,
            )
        )
        print("")
        print("Selection and overlap:")
        print("  Both proposed: {:.1f}".format(stats.get("both_proposed_sum", 0) / steps))
        print("  Truncated:     {:.1f}".format(stats.get("truncated_count_sum", 0) / steps))
        print(
            "  Selected E/S/B: {:.1f} / {:.1f} / {:.1f}".format(
                stats.get("selected_eagle_sum", 0) / steps,
                stats.get("selected_sam_sum", 0) / steps,
                stats.get("selected_both_sum", 0) / steps,
            )
        )
        print("")
        print("SAM quality:")
        print(
            "  Avg match length: {:.1f}".format(
                float(stats.get("sam_match_length_sum", 0.0)) / steps
            )
        )
        print("  Max match length: {}".format(int(stats.get("sam_match_length_max", 0))))
        print(
            "  Skipped: {} / {} ({:.1%})".format(
                int(stats.get("sam_skipped_count", 0)),
                steps,
                int(stats.get("sam_skipped_count", 0)) / steps,
            )
        )

        if accept_rates:
            print("")
            print("Acceptance rates:")
            print("  Eagle: {:.1%}".format(avg_eagle_rate))
            print("  SAM:   {:.1%}".format(avg_sam_rate))
            print("  Delta: {:+.1%}".format(avg_eagle_rate - avg_sam_rate))
            print("")
            print("Total accepted tokens:")
            print("  Eagle: {}".format(total_eagle_accepted))
            print("  SAM:   {}".format(total_sam_accepted))
            print(
                "  SAM contribution: {:.1%}".format(
                    (total_sam_accepted / total_accepted) if total_accepted > 0 else 0.0
                )
            )

        if last:
            print("")
            print("Last step details:")
            print("  SAM avg match length: {:.1f}".format(float(last.get("sam_avg_match_length", 0.0))))
            print("  SAM max match length: {}".format(int(last.get("sam_max_match_length", 0))))
            print("  Both proposed: {}".format(int(last.get("both_proposed_count", 0))))
            print("  Dedup removed: {}".format(int(last.get("dedup_count", 0))))
            print("  Selected E/S/B: {} / {} / {}".format(
                int(last.get("selected_eagle", 0)),
                int(last.get("selected_sam", 0)),
                int(last.get("selected_both", 0)),
            ))
        print("=" * 60)

    def set_cache(self, generation_config: SamdGenerationConfig):
        if self.samd_config.cache_type == "dynamic":
            self.cache = SamdCache(self.lm.config.num_hidden_layers)  # use dynamic cache
        else:
            if self.cache is None:
                print("init static cache...")
                self.cache = SamdStaticCache(
                    self.lm.config,
                    batch_size=1,
                    max_cache_len=generation_config.max_cache_len,
                    device=self.device,
                    dtype=self.dtype,
                    hf_device_map=self.lm.hf_device_map,
                )
            else:
                self.cache.reset()
    
    @torch.inference_mode()
    def generate(self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        generation_config: SamdGenerationConfig = None,
    ) -> Outputs:
        if generation_config is None:
            generation_config = SamdGenerationConfig()
        self.gen_config = generation_config

        assert input_ids.shape[0] == 1, "Only support batch_size == 1"  # [1, N]

        self.set_cache(generation_config)

        self.draft.reset()

        input_ids_list = input_ids.squeeze(0).tolist()
        sample_p = self.prefill(input_ids, attention_mask)

        input_length = input_ids.shape[-1]
        decode_tokens = 0
        decode_steps = 0
        accepet_length_per_step = []
        step_trace: Optional[List[Dict[str, Any]]] = (
            [] if generation_config.collect_diagnosis_trace else None
        )
        t2d_buffer: Optional[torch.Tensor] = None
        if (
            generation_config.collect_diagnosis_trace
            and self.samd_config.tree_method == "eagle3"
        ):
            t2d_buffer = self.draft.tree_model.model.t2d
        for step in range(generation_config.max_new_tokens):
            if input_length + decode_tokens + self.samd_config.max_predicts >= generation_config.max_cache_len:
                break
            sample_p, new_ids, trace_meta = self.decode(sample_p, input_length + decode_tokens)
            if step_trace is not None and trace_meta is not None:
                step_trace.append(make_trace_step(
                    step_idx=decode_steps,
                    path_type=trace_meta["path_type"],
                    accept_length=trace_meta["accept_length"],
                    best_candidate=trace_meta["best_candidate"],
                    candidates=trace_meta["candidates"],
                    tree_logits=trace_meta["tree_logits"],
                    t2d_buffer=t2d_buffer,
                ))
            eos_index = None
            if self.eos_token in new_ids:
                eos_index = new_ids.index(self.eos_token)
                new_ids = new_ids[:eos_index + 1]
            elif self.stop_token is not None and self.stop_token in new_ids:
                eos_index = new_ids.index(self.stop_token)
                new_ids = new_ids[:eos_index + 1]
            input_ids_list.extend(new_ids)
            decode_steps += 1
            decode_tokens += len(new_ids)
            accepet_length_per_step.append(len(new_ids))
            # profile_accept_length("lookup", len(new_ids))
            if eos_index is not None:
                break
            if decode_tokens >= generation_config.max_new_tokens:
                break
        input_ids_list = [input_ids_list[:input_length + generation_config.max_new_tokens]]
        fusion_summary = self.draft.fusion_summary()
        profiler = getattr(generation_config, "fusion_profiler", None)
        profiling = profiler is not None and getattr(profiler, "enabled", False)
        if self.samd_config.fusion_mode == "naive":
            self._print_fusion_analysis()
        elif profiling and fusion_summary is not None and fusion_summary["steps"] > 0:
            print(
                "{}_stats:".format(fusion_summary.get("mode", "tree_fusion")),
                json.dumps(fusion_summary, sort_keys=True),
            )
        return Outputs(
            input_ids_list,
            decode_tokens,
            decode_steps,
            accepet_length_per_step,
            step_trace,
        )

    @torch.inference_mode()
    def stream_generate(self,
        input_ids: torch.Tensor,
        tokenizer: LlamaTokenizer,
        generation_config: SamdGenerationConfig = None, 
    ):
        attention_mask = None
        if generation_config is None:
            generation_config = SamdGenerationConfig()
        self.gen_config = generation_config

        assert input_ids.shape[0] == 1, "Only support batch_size == 1"  # [1, N]

        self.set_cache(generation_config) 

        self.draft.reset()
        
        input_ids_list = input_ids.squeeze(0).tolist()
        sample_p = self.prefill(input_ids, attention_mask)
        
        input_length = input_ids.shape[-1]
        decode_tokens = 0
        for step in range(generation_config.max_steps):
            if input_length + decode_tokens + self.samd_config.max_predicts >= generation_config.max_cache_len:
                break
            sample_p, new_ids, _ = self.decode(sample_p, input_length + decode_tokens)
            eos_index = None
            if self.eos_token in new_ids:
                eos_index = new_ids.index(self.eos_token)
                new_ids = new_ids[:eos_index + 1]
            elif self.stop_token is not None and self.stop_token in new_ids:
                eos_index = new_ids.index(self.stop_token)
                new_ids = new_ids[:eos_index + 1]
            input_ids_list.extend(new_ids)
            yield {
                "text": tokenizer.decode(
                    input_ids_list[input_length:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }
            decode_tokens += len(new_ids)
            if eos_index is not None:
                break
            if decode_tokens >= generation_config.max_new_tokens:
                break
