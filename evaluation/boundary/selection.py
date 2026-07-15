"""Constrained selection and Q2 guard for boundary predictors."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from evaluation.boundary.types import MAX_TRIGGER_RATE, MIN_BOUNDARY_PRECISION


def constrain_selection(
    rows: Sequence[Dict[str, Any]],
    max_trigger_rate: float = MAX_TRIGGER_RATE,
    min_boundary_precision: float = MIN_BOUNDARY_PRECISION,
) -> Dict[str, Any]:
    """Pick the best training row subject to the stage's pass-bar constraints.

    Q2 guard: when no threshold meets the selection constraints, return
    ``selected_by="none"``, ``selection_constraints_met=False``, no selected
    threshold, and a hard-fail sentinel. The unconstrained best is recorded as
    a diagnostic field only and is never used as the selected predictor.
    """
    if not rows:
        raise ValueError("empty sweep")
    eligible = [
        row for row in rows
        if row["trigger_rate"] <= float(max_trigger_rate)
        and row["boundary_precision"] >= float(min_boundary_precision)
    ]
    best_unconstrained = max(
        rows,
        key=lambda row: (
            row["predicted_boundary_oracle_mat"],
            row["boundary_recall"],
            -row["trigger_rate"],
        ),
    )
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["predicted_boundary_oracle_mat"],
                row["boundary_recall"],
                -row["trigger_rate"],
            ),
        )
        selected = dict(selected)
        selected["selection_constraints_met"] = True
        selected["selected_by"] = "constrained"
    else:
        selected = {
            "method": best_unconstrained.get("method"),
            "threshold": None,
            "trigger_rate": None,
            "boundary_precision": None,
            "boundary_recall": None,
            "predicted_boundary_oracle_mat": None,
            "selection_constraints_met": False,
            "selected_by": "none",
        }
    selected["best_unconstrained_threshold"] = best_unconstrained.get("threshold")
    selected["best_unconstrained_trigger_rate"] = best_unconstrained.get("trigger_rate")
    selected["best_unconstrained_oracle_mat"] = best_unconstrained.get(
        "predicted_boundary_oracle_mat"
    )
    return selected


def choose_best(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Default full-stage selection bar (trigger<=15%, precision>=20%)."""
    return constrain_selection(rows, MAX_TRIGGER_RATE, MIN_BOUNDARY_PRECISION)


def selection_rank(row: Dict[str, Any]) -> tuple:
    """Rank selected rows: constrained-pass first, then MAT/recall/-trigger."""
    constrained = 1 if row.get("selection_constraints_met") else 0
    mat = float(row.get("predicted_boundary_oracle_mat", 0.0) or 0.0)
    recall = float(row.get("boundary_recall", 0.0) or 0.0)
    trigger = -float(row.get("trigger_rate", 0.0) or 0.0)
    return (constrained, mat, recall, trigger)
