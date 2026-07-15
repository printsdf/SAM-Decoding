"""Drafter-MARS threshold sweep and selection."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from evaluation.boundary import (
    DEFAULT_NODE_BUDGET,
    Prediction,
    constrain_selection,
    evaluate_predictions,
    run_threshold_sweep,
    selection_rank,
)
from evaluation.drafter_mars.predictors import (
    predict_draft_delta_top_path,
    predict_draft_mars_reachable,
    predict_draft_mars_top_path,
)
from evaluation.drafter_mars.schema import (
    DRAFT_MARS_REACHABLE_QUANTILES,
    DRAFT_MARS_THETA_GRID,
    schema_report,
)
from evaluation.oracle import DecodeStep, select_eagle3_only, select_perfect, selected_mat


def _default_delta_tau_grid(steps: Sequence[DecodeStep]) -> List[float]:
    """Default fixed-logprob-delta grid for the ``draft_delta_top_path`` control.

    Mirrors the theta grid mapped onto a margin scale (tau = -log(1 - theta)).
    """
    return [(-math.log(max(1.0 - float(theta), 1e-6))) for theta in DRAFT_MARS_THETA_GRID]


def _eagle_mat(steps: Sequence[DecodeStep], node_budget: Optional[int]) -> float:
    if not steps:
        return 0.0
    return sum(
        selected_mat(step, select_eagle3_only(step, node_budget)) for step in steps
    ) / float(len(steps))


def _perfect_mat(steps: Sequence[DecodeStep], node_budget: Optional[int]) -> float:
    if not steps:
        return 0.0
    return sum(
        selected_mat(step, select_perfect(step, node_budget)) for step in steps
    ) / float(len(steps))


def _evaluate_heldout(
    valid_steps: Sequence[DecodeStep],
    method: str,
    predictor: Callable[..., Prediction],
    selected: Dict[str, Any],
    node_budget: Optional[int],
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extra_kwargs = extra_kwargs or {}
    threshold = selected.get("threshold")
    if threshold is None:
        return {
            "method": method,
            "threshold": None,
            "node_budget": node_budget,
            "steps": len(valid_steps),
            "triggered_steps": 0,
            "trigger_rate": 0.0,
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "eagle_mat": _eagle_mat(valid_steps, node_budget),
            "perfect_mat": _perfect_mat(valid_steps, node_budget),
            "predicted_boundary_oracle_mat": _eagle_mat(valid_steps, node_budget),
            "oracle_gap_vs_eagle": 0.0,
            "recovered_oracle_ceiling": 0.0,
            "selection_constraints_met": False,
            "selected_by": "none",
            "pass": False,
        }
    predictions = [
        predictor(step, float(threshold), **extra_kwargs) for step in valid_steps
    ]
    metrics = evaluate_predictions(
        valid_steps,
        predictions,
        method=method,
        threshold=float(threshold),
        node_budget=node_budget,
    )
    metrics["selection_constraints_met"] = bool(selected.get("selection_constraints_met"))
    metrics["selected_by"] = selected.get("selected_by")
    return metrics


def _evaluate_heldout_reachable(
    valid_steps: Sequence[DecodeStep],
    selected: Dict[str, Any],
    node_budget: Optional[int],
) -> Dict[str, Any]:
    threshold = selected.get("threshold")
    quantile = selected.get("reachable_quantile")
    if threshold is None or quantile is None:
        return _evaluate_heldout(
            valid_steps,
            "draft_mars_reachable",
            predict_draft_mars_reachable,
            selected,
            node_budget,
        )
    predictions = [
        predict_draft_mars_reachable(step, float(threshold), float(quantile))
        for step in valid_steps
    ]
    metrics = evaluate_predictions(
        valid_steps,
        predictions,
        method="draft_mars_reachable",
        threshold=float(threshold),
        node_budget=node_budget,
    )
    metrics["reachable_quantile"] = float(quantile)
    metrics["selection_constraints_met"] = bool(selected.get("selection_constraints_met"))
    metrics["selected_by"] = selected.get("selected_by")
    return metrics


def calibrate_drafter_mars(
    train_steps: Sequence[DecodeStep],
    valid_steps: Sequence[DecodeStep],
    theta_grid: Sequence[float] = DRAFT_MARS_THETA_GRID,
    reachable_quantiles: Sequence[float] = DRAFT_MARS_REACHABLE_QUANTILES,
    delta_tau_grid: Optional[Sequence[float]] = None,
    node_budget: Optional[int] = DEFAULT_NODE_BUDGET,
    max_trigger_rate: float = 0.15,
    min_boundary_precision: float = 0.20,
) -> Dict[str, Any]:
    """Run the Drafter-MARS threshold sweep + selection on train and heldout.

    Selection maximizes train ``predicted_boundary_oracle_mat`` subject to the
    stage's pass bar (``max_trigger_rate`` / ``min_boundary_precision``). When
    no threshold meets the bar, ``selected_by`` is ``"none"`` and the heldout
    result is a hard fail; the unconstrained best is recorded only as a
    diagnostic field, never as the selected predictor.
    """
    thetas = [float(t) for t in theta_grid]
    if delta_tau_grid is None:
        delta_tau_grid = _default_delta_tau_grid(train_steps)
    taus = [float(t) for t in delta_tau_grid]

    train_sweeps: Dict[str, List[Dict[str, Any]]] = {}
    selected: Dict[str, Dict[str, Any]] = {}

    rows, _best = run_threshold_sweep(
        train_steps,
        "draft_mars_top_path",
        predict_draft_mars_top_path,
        thetas,
        node_budget=node_budget,
    )
    train_sweeps["draft_mars_top_path"] = rows
    selected["draft_mars_top_path"] = constrain_selection(
        rows, max_trigger_rate, min_boundary_precision
    )

    rows, _best = run_threshold_sweep(
        train_steps,
        "draft_delta_top_path",
        predict_draft_delta_top_path,
        taus,
        node_budget=node_budget,
    )
    train_sweeps["draft_delta_top_path"] = rows
    selected["draft_delta_top_path"] = constrain_selection(
        rows, max_trigger_rate, min_boundary_precision
    )

    reachable_train_rows: List[Dict[str, Any]] = []
    reachable_selected: Optional[Dict[str, Any]] = None
    for quantile in reachable_quantiles:
        q = float(quantile)
        rows = []
        for theta in thetas:
            predictions = [
                predict_draft_mars_reachable(step, float(theta), q) for step in train_steps
            ]
            rows.append(
                evaluate_predictions(
                    train_steps,
                    predictions,
                    method="draft_mars_reachable",
                    threshold=float(theta),
                    node_budget=node_budget,
                )
            )
            rows[-1]["reachable_quantile"] = q
        best_q = constrain_selection(rows, max_trigger_rate, min_boundary_precision)
        best_q["reachable_quantile"] = q
        reachable_train_rows.extend(rows)
        if reachable_selected is None or selection_rank(best_q) > selection_rank(reachable_selected):
            reachable_selected = best_q
    train_sweeps["draft_mars_reachable"] = reachable_train_rows
    selected["draft_mars_reachable"] = reachable_selected or {
        "method": "draft_mars_reachable",
        "threshold": None,
        "selected_by": "none",
        "selection_constraints_met": False,
    }

    heldout: Dict[str, Dict[str, Any]] = {}
    for method, predictor in (
        ("draft_mars_top_path", predict_draft_mars_top_path),
        ("draft_delta_top_path", predict_draft_delta_top_path),
    ):
        heldout[method] = _evaluate_heldout(
            valid_steps,
            method,
            predictor,
            selected[method],
            node_budget=node_budget,
        )
    heldout["draft_mars_reachable"] = _evaluate_heldout_reachable(
        valid_steps,
        selected["draft_mars_reachable"],
        node_budget=node_budget,
    )

    return {
        "schema_report": schema_report(list(train_steps) + list(valid_steps)),
        "theta_grid": list(thetas),
        "reachable_quantiles": list(reachable_quantiles),
        "delta_tau_grid": list(taus),
        "node_budget": node_budget,
        "selection_constraints": {
            "max_trigger_rate": float(max_trigger_rate),
            "min_boundary_precision": float(min_boundary_precision),
        },
        "selected_thresholds": selected,
        "train_sweeps": train_sweeps,
        "heldout_metrics": heldout,
    }
