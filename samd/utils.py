import torch
import torch.nn.functional as F
import random
from enum import Enum
from typing import List, Dict, Optional, Callable
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

    if samd_config.fusion_mode == "naive":
        index_dyn, match_dyn = draft.sam_dyn.lookup(start_token)
        index_static, match_static_raw = draft.sam_static.lookup(start_token)
        match_static = match_static_raw - draft.len_bias
        best_match = max(match_dyn, match_static)
        samd_len_threshold = getattr(samd_config, "samd_len_threshold", None)
        if samd_len_threshold is None:
            samd_len_threshold = samd_config.len_threshold

        if best_match < samd_len_threshold:
            eagle_pred_ids, eagle_buffers = draft.tree_model.gen_draft(start_token)
            eagle_nonroot_nodes = max(len(eagle_pred_ids) - 1, 0)
            draft.record_naive_fusion({
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

        eagle_tokens, eagle_buffers, eagle_logprobs = draft.tree_model.gen_draft(
            start_token,
            return_logprobs=True,
        )
        eagle_tree = {
            "tokens": eagle_tokens,
            "logprobs": eagle_logprobs,
            "tree_attn_mask": eagle_buffers["tree_attn_mask"],
            "tree_position_ids": eagle_buffers["tree_position_ids"],
            "tree_retrieve_indices": eagle_buffers["tree_retrieve_indices"],
        }

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

        fused_tree = fuse_eagle_sam_naive(
            eagle_tree=eagle_tree,
            sam_candidates=sam_candidates,
            start_token=start_token,
            config=samd_config.fusion_config,
            sam_match_length=sam_match_length,
            eagle_logprobs=eagle_logprobs,
        )
        fused_tree.metadata.setdefault("sam_skipped", False)
        draft.record_naive_fusion(fused_tree.metadata)
        tokens = fused_tree.tokens.unsqueeze(0)
        tokens_ext = torch.cat(
            [fused_tree.tokens, torch.zeros(1, dtype=torch.long, device=fused_tree.tokens.device)]
        )
        candidate_tokens = tokens_ext[fused_tree.retrieve_indices]
        return Candidates(
            CandidateType.tree,
            tokens,
            candidate_tokens,
            fused_tree.buffers_kwargs,
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
