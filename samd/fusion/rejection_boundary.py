"""Rejection-boundary fusion for EAGLE + SAM.

Phase 1: Root-level confidence gating (prototype).
"""

import torch
from typing import Optional


def compute_confidence_margin(logprobs: torch.Tensor, depth: int = 0) -> float:
    """Compute confidence margin (top1 - top2 logprob) at given depth.

    Args:
        logprobs: EAGLE draft logprobs, typically shape [num_nodes] or [num_nodes, vocab_size]
        depth: Tree depth to check (0 = root)

    Returns:
        margin: top1_logprob - top2_logprob. Higher = more confident.
                Returns 1.0 if insufficient candidates for margin calculation.
    """
    if logprobs is None or len(logprobs) == 0:
        return 1.0  # No logprobs available, assume high confidence (skip SAM)

    # Handle different logprob formats
    if logprobs.dim() == 1:
        # Single logprob per node (already selected)
        # Cannot compute margin, return conservative high value
        return 1.0

    if logprobs.dim() == 2:
        # [num_nodes, vocab_size] or [num_nodes, top_k]
        if depth >= logprobs.shape[0]:
            return 1.0  # Depth out of range

        depth_logprobs = logprobs[depth]

        # Get top-2 values
        if depth_logprobs.numel() < 2:
            return 1.0

        top2_values, _ = torch.topk(depth_logprobs, k=min(2, depth_logprobs.numel()))

        if len(top2_values) < 2:
            return 1.0

        margin = (top2_values[0] - top2_values[1]).item()
        return margin

    # Unknown format, default to high confidence
    return 1.0


def should_trigger_sam(margin: float, threshold: float) -> bool:
    """Decision function: trigger SAM if confidence margin > threshold.

    High margin = high confidence gap = model is "confidently wrong"
    This is where SAM rescue is most valuable.

    Args:
        margin: Confidence margin from compute_confidence_margin()
        threshold: Trigger threshold (higher = more selective SAM usage)

    Returns:
        True if SAM should be activated, False otherwise
    """
    return margin > threshold
