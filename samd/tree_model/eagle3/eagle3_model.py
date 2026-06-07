"""EAGLE3 draft model.

Direct port of ../EAGLE/eagle/model/cnets.py:Model with these adaptations:
- Removed ``load_emb`` / ``path`` parameters and ``hf_hub_download`` dependency
  (samd is inference-only; embed_tokens initialization comes from the checkpoint).
- ``load_weight`` accepts a directory and tries ``pytorch_model.bin`` then
  ``model.safetensors``; uses ``strict=False`` (per design D4) so the
  ``vocab_size == draft_vocab_size`` simplification (official code deletes
  ``d2t``/``t2d``) loads cleanly. Missing keys outside ``{d2t, t2d}`` and any
  unexpected keys raise a warning print.
- Loading prints one diagnostic line containing ``missing_keys`` /
  ``unexpected_keys`` / ``draft_vocab_size`` / ``vocab_size`` / ``hidden_size`` /
  tree generation parameters / captured layer indices, per design 2.2
  observability constraint.

Author note: building blocks live in eagle3_utils.py (LlamaAttention / MLP /
RMSNorm / LlamaDecoderLayeremb) — those are the EAGLE3-specific structure
(q/k/v_proj input dim = hidden_size * 2; midlayer concats input_emb + hidden).
"""
import math
import os
import time
from typing import List, Optional

import torch
from torch import nn

from .eagle3_utils import (
    LlamaDecoderLayeremb,
    LlamaRMSNorm,
    _expand_mask,
    _make_causal_mask,
)


CAPTURED_LAYER_INDICES_DOC = "captured_in_base_patch={idx==2, idx==N//2, idx==N-3}"


class Eagle3Model(nn.Module):

    def __init__(
        self,
        config,
        total_tokens=63,
        depth=5,
        top_k=8,
        threshold=1.0,
    ):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)

        self.midlayer = LlamaDecoderLayeremb(config)

        if hasattr(config, "target_hidden_size"):
            self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)
        else:
            self.fc = nn.Linear(config.hidden_size * 3, self.hidden_size, bias=False)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        d2t = torch.zeros((config.draft_vocab_size,), dtype=torch.long)
        t2d = torch.ones((config.vocab_size,), dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
        self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None

    def reset_kv(self):
        self.stable_kv = None

    def load_weight(self, path: str):
        print("load eagle3 model from {}...".format(path))
        start = time.perf_counter()
        bin_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            safetensors_path = os.path.join(path, "model.safetensors")
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(
                    "neither pytorch_model.bin nor model.safetensors found under {}".format(path)
                )
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)

        load_result = self.load_state_dict(state_dict, strict=False)
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        print(
            "eagle3 load: missing_keys={}, unexpected_keys={}, "
            "draft_vocab_size={}, vocab_size={}, hidden_size={}, "
            "total_token={}, depth={}, top_k={}, {}".format(
                missing, unexpected,
                self.config.draft_vocab_size, self.vocab_size, self.hidden_size,
                self.total_tokens + 1, self.depth, self.top_k,
                CAPTURED_LAYER_INDICES_DOC,
            )
        )
        # Stricter than blanket strict=False: only allow keys we know are
        # supplied/derived elsewhere.
        allowed_missing = {"embed_tokens.weight"}  # Eagle3 integration class copies from base lm
        if self.config.vocab_size == self.config.draft_vocab_size:
            # When draft vocab matches base vocab, official checkpoints drop
            # d2t/t2d (no remapping needed); d2t zeros and t2d all-True preserve
            # identity reachability for diagnosis and any later sidecar checks.
            allowed_missing.update({"d2t", "t2d"})
        real_missing = [k for k in missing if k not in allowed_missing]
        if real_missing:
            raise RuntimeError(
                "eagle3 load_weight: required state_dict keys missing: {}. "
                "Check the checkpoint matches Eagle3Config "
                "(draft_vocab_size={}, vocab_size={}, hidden_size={}).".format(
                    real_missing,
                    self.config.draft_vocab_size, self.vocab_size, self.hidden_size,
                )
            )
        if unexpected:
            print("WARNING eagle3: unexpected keys present in checkpoint: {}".format(unexpected))

        end = time.perf_counter()
        print("loading ended in {} seconds.".format(end - start))

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
            ] = torch.finfo(torch.float32).min

        return combined_attention_mask

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        with torch.no_grad():
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        inputs_embeds = inputs_embeds.to(hidden_states.dtype)
        if hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        next_decoder_cache = () if use_cache else None
        past_key_value = past_key_values[0] if past_key_values is not None else None
        layer_outputs = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=True,
        )
        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        hidden_states = layer_outputs[0]

        if use_cache:
            return hidden_states, next_decoder_cache
        return hidden_states

    @torch.no_grad()
    def topK_genrate(self, hidden_states, input_ids, head, logits_processor=None):
        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        logprobs_list = []
        parents_list = []
        ss_token = []

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        # cnets's incremental forward: when stable_kv is available, slice input_ids
        # to its delta tail so the new KV computed here matches the delta-length
        # hidden_states the caller passed. Skipping this branch (e.g. by always
        # running full-length) loses the EAGLE3 speedup.
        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            kv_len = self.stable_kv[0][0].shape[-2]
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids[:, kv_len:],
                past_key_values=self.stable_kv,
                use_cache=True,
            )
        else:
            out_hidden, past_key_values = self(
                hidden_states, input_ids=input_ids, use_cache=True
            )
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = self.lm_head(self.norm(last_hidden))

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        logprobs_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index)
            input_ids = topk_index
        else:
            ss_token.append(topk_index + self.d2t[topk_index])
            input_ids = topk_index + self.d2t[topk_index]
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            out_hidden, past_key_values = self(
                input_hidden,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            last_headout = self.lm_head(self.norm(out_hidden[0]))
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]

            input_ids = topk_index.view(-1)[topk_cs_index][None]

            if self.config.vocab_size == self.config.draft_vocab_size:
                ss_token.append(topk_index)
            else:
                input_ids = input_ids + self.d2t[input_ids]
                ss_token.append(topk_index + self.d2t[topk_index])
            scores_list.append(cu_scores)
            logprobs_list.append(topk_p)
            tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        logprobs_list = torch.cat(logprobs_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)
        draft_logprobs = logprobs_list[top_scores_index]
        draft_logprobs = torch.cat(
            (
                torch.zeros(1, dtype=draft_logprobs.dtype, device=draft_logprobs.device),
                draft_logprobs,
            ),
            dim=0,
        )

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()

        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]
        draft_logprobs = draft_logprobs[None]

        del parents_list, scores_list, logprobs_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long).to(hidden_states.device)
        del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
        tree_position_ids = tree_position_ids[None].to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, draft_logprobs
