"""Default-suite boundary predictor calibration (low_margin + random)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evaluation.boundary.candidates import (
    eagle_candidates,
    eligible_margin_candidates,
    question_index,
)
from evaluation.boundary.low_margin import predict_low_margin_node
from evaluation.boundary.metrics import evaluate_predictions
from evaluation.boundary.selection import constrain_selection
from evaluation.boundary.sweep import run_threshold_sweep, threshold_grid
from evaluation.boundary.types import (
    DEFAULT_MAX_SWEEP_THRESHOLDS,
    DEFAULT_NODE_BUDGET,
    MAX_TRIGGER_RATE,
    MIN_BOUNDARY_PRECISION,
    Prediction,
)
from evaluation.oracle import DecodeStep, select_eagle3_only, select_perfect, selected_mat


def warn(message: str) -> None:
    print("WARNING: {}".format(message), file=sys.stderr)


def schema_report(steps: Sequence[DecodeStep]) -> Dict[str, Any]:
    total = len(steps)
    labeled = sum(1 for step in steps if step.has_rejection_boundary_label)
    question_indexed = sum(1 for step in steps if question_index(step) is not None)
    eagle = [
        candidate
        for step in steps
        for candidate in eagle_candidates(step)
    ]
    multi_sibling = [
        candidate
        for candidate in eagle
        if candidate.sibling_margin is not None
    ]
    report = {
        "steps": total,
        "steps_with_rejection_boundary_label": labeled,
        "steps_with_question_index": question_indexed,
        "eagle_candidates": len(eagle),
        "eagle_candidates_with_tree_index": sum(
            1 for candidate in eagle if candidate.tree_index is not None
        ),
        "eagle_candidates_with_parent_index": sum(
            1 for candidate in eagle if candidate.parent_index is not None
        ),
        "eagle_candidates_with_local_logprob": sum(
            1 for candidate in eagle if candidate.local_logprob is not None
        ),
        "eagle_candidates_with_sibling_margin": len(multi_sibling),
        "eagle_candidates_with_cumulative_path_logprob": sum(
            1 for candidate in eagle
            if candidate.cumulative_path_logprob is not None
        ),
    }
    report["schema_ok"] = (
        total > 0
        and labeled == total
        and question_indexed == total
        and len(eagle) > 0
        and report["eagle_candidates_with_tree_index"] > 0
        and report["eagle_candidates_with_local_logprob"] > 0
        and report["eagle_candidates_with_sibling_margin"] > 0
    )
    return report


def require_calibration_schema(steps: Sequence[DecodeStep]) -> Dict[str, Any]:
    report = schema_report(steps)
    if not report["schema_ok"]:
        raise ValueError(
            "trace is missing required boundary-calibration schema fields: {}".format(
                json.dumps(report, sort_keys=True)
            )
        )
    return report


def split_steps(
    steps: Sequence[DecodeStep],
    begin: int,
    end: int,
) -> List[DecodeStep]:
    selected = [
        step
        for step in steps
        if question_index(step) is not None
        and int(begin) <= int(question_index(step)) < int(end)
    ]
    if not selected:
        raise ValueError("no decode steps found for question split [{}, {})".format(begin, end))
    return selected


def matched_random_predictions(
    steps: Sequence[DecodeStep],
    target_trigger_rate: float,
    seed: int,
) -> List[Prediction]:
    rng = random.Random(int(seed))
    predictions: List[Prediction] = []
    for step in steps:
        candidates = eagle_candidates(step)
        if candidates and rng.random() < float(target_trigger_rate):
            candidate = rng.choice(candidates)
            predictions.append(
                Prediction(
                    method="random_node",
                    candidate=candidate,
                    depth=int(candidate.depth),
                    score=None,
                    threshold=float(target_trigger_rate),
                )
            )
        else:
            predictions.append(
                Prediction("random_node", None, None, threshold=float(target_trigger_rate))
            )
    return predictions


def average_metric_rows(rows: Sequence[Dict[str, Any]], method: str) -> Dict[str, Any]:
    if not rows:
        raise ValueError("cannot average empty rows")
    result: Dict[str, Any] = {"method": method, "seeds": len(rows)}
    keys = set().union(*(row.keys() for row in rows))
    for key in sorted(keys):
        if key in ("method",):
            continue
        values = [
            row.get(key)
            for row in rows
            if isinstance(row.get(key), (int, float)) and row.get(key) is not None
        ]
        if values:
            result[key] = sum(float(value) for value in values) / float(len(values))
    return result


def hard_fail_heldout(
    valid_steps: Sequence[DecodeStep],
    method: str,
    node_budget: Optional[int],
) -> Dict[str, Any]:
    eagle_mat = sum(
        selected_mat(step, select_eagle3_only(step, node_budget))
        for step in valid_steps
    ) / float(len(valid_steps)) if valid_steps else 0.0
    perfect_mat = sum(
        selected_mat(step, select_perfect(step, node_budget))
        for step in valid_steps
    ) / float(len(valid_steps)) if valid_steps else 0.0
    return {
        "method": method,
        "threshold": None,
        "node_budget": node_budget,
        "steps": len(valid_steps),
        "triggered_steps": 0,
        "trigger_rate": 0.0,
        "boundary_precision": 0.0,
        "boundary_recall": 0.0,
        "eagle_mat": eagle_mat,
        "perfect_mat": perfect_mat,
        "predicted_boundary_oracle_mat": eagle_mat,
        "oracle_gap_vs_eagle": 0.0,
        "recovered_oracle_ceiling": 0.0,
        "selection_constraints_met": False,
        "selected_by": "none",
    }


def calibrate(
    steps: Sequence[DecodeStep],
    train_begin: int,
    train_end: int,
    valid_begin: int,
    valid_end: int,
    max_thresholds: int = DEFAULT_MAX_SWEEP_THRESHOLDS,
    node_budget: Optional[int] = DEFAULT_NODE_BUDGET,
    predictor_suite: str = "default",
    theta_grid: Optional[Sequence[float]] = None,
    reachable_quantiles: Optional[Sequence[float]] = None,
    require_raw_draft_logits: bool = False,
    stage: str = "full",
) -> Dict[str, Any]:
    schema = require_calibration_schema(steps)
    train_steps = split_steps(steps, train_begin, train_end)
    valid_steps = split_steps(steps, valid_begin, valid_end)

    if stage == "smoke":
        max_trigger_rate = 0.20
        min_boundary_precision = 0.15
    else:
        max_trigger_rate = MAX_TRIGGER_RATE
        min_boundary_precision = MIN_BOUNDARY_PRECISION

    margin_values = [
        float(candidate.sibling_margin)
        for step in train_steps
        for candidate in eligible_margin_candidates(step)
    ]
    margin_thresholds = threshold_grid(margin_values, max_thresholds=max_thresholds)
    if not margin_thresholds:
        raise ValueError("no sibling_margin values available for threshold sweep")

    train_sweeps: Dict[str, List[Dict[str, Any]]] = {}
    selected: Dict[str, Dict[str, Any]] = {}
    rows, _best = run_threshold_sweep(
        train_steps,
        "low_margin_node",
        predict_low_margin_node,
        margin_thresholds,
        node_budget=node_budget,
    )
    train_sweeps["low_margin_node"] = rows
    selected["low_margin_node"] = constrain_selection(
        rows, max_trigger_rate, min_boundary_precision
    )

    low_margin_trigger_rate = (
        selected["low_margin_node"].get("trigger_rate")
        or rows[0].get("trigger_rate", 0.0)
        if rows
        else 0.0
    )
    random_rows = []
    random_valid_rows = []
    for seed in range(10):
        random_rows.append(
            evaluate_predictions(
                train_steps,
                matched_random_predictions(train_steps, low_margin_trigger_rate, seed),
                method="random_node",
                threshold=low_margin_trigger_rate,
                node_budget=node_budget,
            )
        )
        random_valid_rows.append(
            evaluate_predictions(
                valid_steps,
                matched_random_predictions(valid_steps, low_margin_trigger_rate, seed),
                method="random_node",
                threshold=low_margin_trigger_rate,
                node_budget=node_budget,
            )
        )
    selected["random_node"] = average_metric_rows(random_rows, "random_node")

    heldout: Dict[str, Dict[str, Any]] = {}
    low_threshold = selected["low_margin_node"].get("threshold")
    if low_threshold is None:
        heldout["low_margin_node"] = hard_fail_heldout(
            valid_steps, "low_margin_node", node_budget
        )
    else:
        heldout["low_margin_node"] = evaluate_predictions(
            valid_steps,
            [predict_low_margin_node(step, float(low_threshold)) for step in valid_steps],
            method="low_margin_node",
            threshold=float(low_threshold),
            node_budget=node_budget,
        )
    heldout["random_node"] = average_metric_rows(random_valid_rows, "random_node")

    result: Dict[str, Any] = {
        "schema_report": schema,
        "splits": {
            "train": {"begin": train_begin, "end": train_end, "steps": len(train_steps)},
            "heldout": {"begin": valid_begin, "end": valid_end, "steps": len(valid_steps)},
        },
        "threshold_grid": {
            "max_thresholds": int(max_thresholds),
            "margin_values": len(margin_values),
            "margin_thresholds": len(margin_thresholds),
        },
        "node_budget": node_budget,
        "selected_thresholds": selected,
        "train_sweeps": train_sweeps,
        "heldout_metrics": heldout,
    }

    if predictor_suite in ("drafter_mars", "all"):
        from evaluation.drafter_mars import (
            DRAFT_MARS_REACHABLE_QUANTILES,
            DRAFT_MARS_THETA_GRID,
            calibrate_drafter_mars,
            require_drafter_mars_schema,
        )

        thetas = list(theta_grid) if theta_grid is not None else list(DRAFT_MARS_THETA_GRID)
        quants = (
            list(reachable_quantiles)
            if reachable_quantiles is not None
            else list(DRAFT_MARS_REACHABLE_QUANTILES)
        )
        if require_raw_draft_logits:
            require_drafter_mars_schema(list(train_steps) + list(valid_steps))
        mars = calibrate_drafter_mars(
            train_steps,
            valid_steps,
            theta_grid=thetas,
            reachable_quantiles=quants,
            node_budget=node_budget,
            max_trigger_rate=max_trigger_rate,
            min_boundary_precision=min_boundary_precision,
        )
        result["drafter_mars"] = mars
        result["selection_constraints"] = {
            "max_trigger_rate": float(max_trigger_rate),
            "min_boundary_precision": float(min_boundary_precision),
            "stage": stage,
        }

    gates = {
        "trigger_rate_lte_bar": heldout["low_margin_node"]["trigger_rate"] <= max_trigger_rate,
        "boundary_precision_gte_bar": (
            heldout["low_margin_node"]["boundary_precision"] >= min_boundary_precision
        ),
        "oracle_mat_gte_eagle": (
            heldout["low_margin_node"]["predicted_boundary_oracle_mat"]
            >= heldout["low_margin_node"]["eagle_mat"] * 1.01
        ),
        "low_margin_beats_random": (
            heldout["low_margin_node"]["predicted_boundary_oracle_mat"]
            > heldout["random_node"].get("predicted_boundary_oracle_mat", 0.0)
        ),
    }
    result["decision_gates"] = gates
    result["pass"] = all(gates.values())
    return result


def write_outputs(result: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "schema_report.json": result["schema_report"],
        "train_sweep_q0_82.json": result["train_sweeps"],
        "heldout_metrics_q82_164.json": result["heldout_metrics"],
        "selected_thresholds.json": result["selected_thresholds"],
        "boundary_predictor_calibration_summary.json": result,
    }
    for name, payload in payloads.items():
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
