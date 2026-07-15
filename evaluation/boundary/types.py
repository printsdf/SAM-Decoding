"""Boundary-predictor shared types and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evaluation.oracle import OracleCandidate


MAX_TRIGGER_RATE = 0.15
MIN_BOUNDARY_PRECISION = 0.20
DEFAULT_MAX_SWEEP_THRESHOLDS = 256
DEFAULT_NODE_BUDGET = 60


@dataclass(frozen=True)
class Prediction:
    method: str
    candidate: Optional[OracleCandidate]
    depth: Optional[int]
    score: Optional[float] = None
    threshold: Optional[float] = None


def validate_probability_threshold(value: float) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError("probability threshold must be in [0, 1]: {}".format(value))
    return value
