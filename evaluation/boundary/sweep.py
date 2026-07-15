"""Threshold grids and sweeps for boundary predictors."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.boundary.metrics import evaluate_predictions
from evaluation.boundary.selection import choose_best
from evaluation.boundary.types import DEFAULT_MAX_SWEEP_THRESHOLDS, DEFAULT_NODE_BUDGET, Prediction
from evaluation.oracle import DecodeStep


def threshold_grid(
    values: Iterable[float],
    max_thresholds: int = DEFAULT_MAX_SWEEP_THRESHOLDS,
) -> List[float]:
    unique = sorted({float(value) for value in values if math.isfinite(float(value))})
    if not unique:
        return []
    max_thresholds = max(int(max_thresholds), 2)
    if len(unique) == 1:
        value = unique[0]
        epsilon = max(abs(value) * 1e-6, 1e-6)
        return [value - epsilon, value + epsilon]
    all_thresholds = [unique[0] - max(abs(unique[0]) * 1e-6, 1e-6)]
    all_thresholds.extend((left + right) / 2.0 for left, right in zip(unique, unique[1:]))
    all_thresholds.append(unique[-1] + max(abs(unique[-1]) * 1e-6, 1e-6))
    if len(all_thresholds) <= max_thresholds:
        return all_thresholds
    indexes = {
        int(round(index * (len(all_thresholds) - 1) / float(max_thresholds - 1)))
        for index in range(max_thresholds)
    }
    return [all_thresholds[index] for index in sorted(indexes)]


def run_threshold_sweep(
    steps: Sequence[DecodeStep],
    method: str,
    predictor: Callable[[DecodeStep, float], Prediction],
    thresholds: Sequence[float],
    node_budget: Optional[int] = DEFAULT_NODE_BUDGET,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        predictions = [predictor(step, float(threshold)) for step in steps]
        rows.append(
            evaluate_predictions(
                steps,
                predictions,
                method=method,
                threshold=float(threshold),
                node_budget=node_budget,
            )
        )
    return rows, choose_best(rows)
