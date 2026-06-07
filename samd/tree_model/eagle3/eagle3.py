import torch
from typing import Dict, List, Tuple, Union

from transformers import LlamaForCausalLM

from ...samd_config import SamdConfig
from ..tree import TreeModel
from .eagle3_config import Eagle3Config
from .eagle3_model import Eagle3Model
from .tail_sidecar import attach_tail_sidecar


class Eagle3(TreeModel):

    def __init__(
        self,
        config: SamdConfig,
        lm: LlamaForCausalLM,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        # Kept for interface symmetry with Eagle2; EAGLE3 uses its own self.lm_head + self.norm internally.
        self.head: torch.nn.Linear = lm.lm_head
        self.model: Eagle3Model = Eagle3Model(
            config=Eagle3Config(**config.tree_config),
            total_tokens=config.eagle3_total_token,
            depth=config.eagle3_depth,
            top_k=config.eagle3_top_k,
        ).to(device=device, dtype=dtype)
        self.model.load_weight(config.tree_model_path)
        # EAGLE3 official checkpoints omit embed_tokens.weight (the draft shares
        # the base model's embedding; load_emb=True in cnets.py:488-519 fills it
        # from base path at training time). Copy from the loaded base model.
        with torch.no_grad():
            base_emb = lm.model.embed_tokens.weight.data
            draft_emb = self.model.embed_tokens.weight.data
            if base_emb.shape == draft_emb.shape:
                draft_emb.copy_(base_emb)
            else:
                print(
                    "WARNING eagle3: base/draft embed_tokens shape mismatch "
                    "(base {} vs draft {}); skipping copy, draft may produce "
                    "garbage tokens".format(tuple(base_emb.shape), tuple(draft_emb.shape))
                )
        if config.eagle3_tail_path is not None:
            attach_tail_sidecar(
                self.model,
                tail_path=config.eagle3_tail_path,
                tail_type=config.eagle3_tail_type,
                dtype=dtype,
                device=device,
            )
        else:
            self.model.init_tree()

        # State machine: cumulative_tokens grows over the whole generation (cleared
        # only by reset); pending_hidden_states is the delta since the last gen_draft
        # (cleared inside gen_draft). Together they restore the EAGLE3 cnets.py
        # incremental convention on top of samd's accumulating update().
        self.cumulative_tokens: torch.Tensor = None
        self.pending_hidden_states: torch.Tensor = None

    def reset(self):
        self.model.stable_kv = None
        self.cumulative_tokens = None
        self.pending_hidden_states = None

    def update(
        self,
        tokens: torch.Tensor,
        last_hidden_states: torch.Tensor,
        **kwargs,
    ):
        tokens = tokens.to(self.device)
        if self.cumulative_tokens is None:
            self.cumulative_tokens = tokens
        else:
            self.cumulative_tokens = torch.cat([self.cumulative_tokens, tokens], dim=-1)
        if self.pending_hidden_states is None:
            self.pending_hidden_states = last_hidden_states
        else:
            self.pending_hidden_states = torch.cat(
                [self.pending_hidden_states, last_hidden_states], dim=-2
            )

    def gen_draft(
        self,
        start_token: int,
        return_logprobs: bool = False,
    ) -> Union[
        Tuple[List[int], Dict[str, torch.Tensor]],
        Tuple[List[int], Dict[str, torch.Tensor], List[float]],
    ]:
        start_token = torch.tensor([start_token], dtype=torch.long, device=self.device)
        # input_ids carries the full history (cumulative_tokens + start_token);
        # hidden_states carries only the delta since the last gen_draft. cnets's
        # `input_ids[:, kv_len:]` slice will match the delta length downstream.
        input_ids_full = torch.cat((self.cumulative_tokens, start_token), dim=-1)
        pending = self.pending_hidden_states
        self.pending_hidden_states = None
        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            draft_logprobs,
        ) = self.model.topK_genrate(
            pending[None],
            input_ids_full[None],
            self.head,
        )
        pred_ids = draft_tokens.view(-1).tolist()
        buffers_kwargs = {
            "tree_attn_mask": tree_mask,
            "tree_position_ids": tree_position_ids,
            "tree_retrieve_indices": retrieve_indices,
        }
        if return_logprobs:
            logprobs = [float(value) for value in draft_logprobs.view(-1).tolist()]
            return pred_ids, buffers_kwargs, logprobs
        return pred_ids, buffers_kwargs

    def gen_buffers(self):
        return {
            "tree_attn_mask": None,
            "tree_position_ids": None,
            "tree_retrieve_indices": None,
        }
