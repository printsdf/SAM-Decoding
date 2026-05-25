"""Per-step diagnosis trace utilities for samd.

Records what happened at each decode step so downstream analysis can compute
V_miss (verifier-miss rate) and accept-length statistics without intercepting
the SamdModel main path at run time. Decoupled from concrete tree_method via
the optional ``t2d_buffer`` parameter so non-eagle3 callers degrade cleanly
to reachable=None.

The trace dict shape is the 7-key schema described in
``.codestable/features/2026-05-24-medqa-vmiss-eval/medqa-vmiss-eval-design.md``
sections 0 and 2.1, kept stable so ``evaluation/analyze_vmiss.py`` can rely on
it.
"""
from typing import Any, Dict, Optional

import torch


def _token_reachable(
    token_id: Optional[int],
    t2d: Optional[torch.Tensor],
) -> Optional[bool]:
    """Look up whether ``token_id`` is covered by the draft model vocabulary.

    Returns ``None`` when ``t2d`` is unavailable (non-eagle3 tree_method) or
    when ``token_id`` is ``None`` / out of range; otherwise the boolean from
    ``t2d``. Mirrors
    ``../EAGLE/eagle/model/ea_model.py:_tokens_reachable_by_draft_vocab``.
    """
    if token_id is None or t2d is None:
        return None
    if token_id < 0 or token_id >= t2d.shape[0]:
        return None
    return bool(t2d[token_id].item())


def make_trace_step(
    step_idx: int,
    path_type: str,
    accept_length: int,
    best_candidate: Optional[int] = None,
    candidates: Optional[torch.Tensor] = None,
    tree_logits: Optional[torch.Tensor] = None,
    t2d_buffer: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Build a per-step trace entry with seven fixed keys.

    For sequence path / accept-to-end / non-eagle3 tree_method, the four
    ``*_token_id`` and ``*_reachable`` fields default to ``None`` per design D3.

    Args:
        step_idx: 0-indexed decode step.
        path_type: ``"sequence"`` or ``"tree"`` (matches
            ``samd/draft.py:CandidateType.value``).
        accept_length: number of tokens accepted this step in the samd
            convention (includes the +1 base token, so accept_length >= 1).
        best_candidate: row index into ``candidates``; tree path only.
        candidates: ``[num_paths, max_depth+1]`` verified candidate tokens;
            tree path only.
        tree_logits: ``[num_paths, max_depth+1, vocab]`` verifier logits over
            candidate positions; tree path only.
        t2d_buffer: bool ``[base_vocab_size]`` from ``Eagle3Model.t2d``;
            eagle3 only.

    Returns:
        Dict with these keys (in this order):
            step_idx, path_type, accept_length,
            first_rejected_token_id, first_rejected_reachable,
            verifier_target_token_id, verifier_target_reachable.
    """
    accept_length = int(accept_length)
    first_rejected: Optional[int] = None
    verifier_target: Optional[int] = None

    if (
        path_type == "tree"
        and candidates is not None
        and best_candidate is not None
        and accept_length < candidates.shape[1]
    ):
        rejected_val = int(
            candidates[best_candidate, accept_length].detach().cpu().item()
        )
        if rejected_val >= 0:
            first_rejected = rejected_val

        # samd's eval_posterior returns accept_length = #-accepted (includes
        # the +1 base token). ``candidates[:, accept_length]`` is rejected;
        # the verifier's logits that predicted this position is
        # ``logits[:, accept_length - 1]`` because ``logits[i]`` predicts
        # ``candidates[i + 1]``. Guard accept_length - 1 >= 0.
        if (
            tree_logits is not None
            and 0 <= accept_length - 1 < tree_logits.shape[1]
        ):
            verifier_target = int(
                torch.argmax(tree_logits[best_candidate, accept_length - 1])
                .detach()
                .cpu()
                .item()
            )

    return {
        "step_idx": int(step_idx),
        "path_type": path_type,
        "accept_length": accept_length,
        "first_rejected_token_id": first_rejected,
        "first_rejected_reachable": _token_reachable(first_rejected, t2d_buffer),
        "verifier_target_token_id": verifier_target,
        "verifier_target_reachable": _token_reachable(verifier_target, t2d_buffer),
    }
