"""EAGLE3-specific patches for LlamaForCausalLM / LlamaModel.

LlamaModel.forward is a near-verbatim copy of transformers 4.46.x
``LlamaModel.forward`` with one modification: it always collects hidden_states
at layer indices ``{2, N//2, N-3}`` and packs them into the output's
``hidden_states`` slot (overriding the standard usage). This matches
../EAGLE/eagle/model/modeling_llama_kv.py:1137-1139 (the EAGLE3 fork's pattern)
which is how the official EAGLE3 training and inference path captures the
low/mid/high features. Using monkey patch instead of fork keeps samd's
existing patch architecture intact.

LlamaForCausalLM.forward then concatenates those 3 captured tensors along the
last dim and exposes the result as ``SamdCausalLMOutputWithPast.last_hidden_states``
(shape ``[B, S, 3H]``).

``_update_causal_mask`` is reused from ``samd.model_patch.llama`` — tree_mask
injection logic is identical between eagle2 and eagle3 paths.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    Cache,
    DynamicCache,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from .llama import SamdCausalLMOutputWithPast, _update_causal_mask


EAGLE3_TARGET_LAYER_OFFSETS = (2, "N//2", "N-3")  # documentation only


def llama_model_forward_eagle3(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = DynamicCache()
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # EAGLE3: always-on collection of layer-input hidden_states at the three
    # canonical indices. Matches modeling_llama_kv.py:1137-1139.
    eagle3_captured: List[torch.Tensor] = []
    n_layers = len(self.layers)
    eagle3_targets = {2, n_layers // 2, n_layers - 3}

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for idx, decoder_layer in enumerate(self.layers):
        if idx in eagle3_targets:
            eagle3_captured.append(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    # Pack the 3 captured hidden_states into output.hidden_states slot, matching
    # modeling_llama_kv.py's pattern of overwriting that field with the EAGLE3
    # captures regardless of output_hidden_states.
    eagle3_hidden_states = tuple(eagle3_captured)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, next_cache, eagle3_hidden_states, all_self_attns] if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=eagle3_hidden_states,
        attentions=all_self_attns,
    )


def llama_for_causal_lm_forward_eagle3(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **loss_kwargs,
) -> Union[Tuple, SamdCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    # EAGLE3: our patched LlamaModel.forward always packs the 3 captured
    # layer-input hidden_states into outputs.hidden_states. If it's missing
    # the eagle3 base-model patch wasn't installed correctly.
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        captured = outputs.hidden_states
    else:
        raise RuntimeError(
            "LlamaModel.forward did not emit EAGLE3 captured hidden_states; "
            "ensure eagle3_attn_patch_dict was applied to LlamaModel."
        )
    last_hidden_states_3h = torch.cat(captured, dim=-1)

    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return SamdCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        last_hidden_states=last_hidden_states_3h,
        past_key_values=outputs.past_key_values,
        hidden_states=None,
        attentions=outputs.attentions,
    )


llama_eagle3_patch_dict = {
    LlamaForCausalLM: [("forward", llama_for_causal_lm_forward_eagle3)],
}

llama_eagle3_attn_patch_dict = {
    LlamaModel: [
        ("_update_causal_mask", _update_causal_mask),
        ("forward", llama_model_forward_eagle3),
    ],
}
