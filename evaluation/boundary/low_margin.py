"""Unconditioned low-margin baseline predictor."""

from __future__ import annotations

from evaluation.boundary.candidates import eligible_margin_candidates
from evaluation.boundary.types import Prediction
from evaluation.oracle import DecodeStep


def predict_low_margin_node(step: DecodeStep, threshold: float) -> Prediction:
    for candidate in eligible_margin_candidates(step):
        margin = float(candidate.sibling_margin)
        if margin < float(threshold):
            return Prediction(
                method="low_margin_node",
                candidate=candidate,
                depth=int(candidate.depth),
                score=margin,
                threshold=float(threshold),
            )
    return Prediction("low_margin_node", None, None, threshold=float(threshold))
