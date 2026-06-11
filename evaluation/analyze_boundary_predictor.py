#!/usr/bin/env python3
"""Offline calibration for EAGLE rejection-boundary predictors."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.oracle_fusion_analysis import (
    DecodeStep,
    OracleCandidate,
    candidate_matches_acceptance,
    candidate_nonroot_path,
    load_decode_traces,
    select_eagle3_only,
    select_perfect,
    selected_mat,
)


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


def warn(message: str) -> None:
    print("WARNING: {}".format(message), file=sys.stderr)


def validate_probability_threshold(value: float) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError("probability threshold must be in [0, 1]: {}".format(value))
    return value


def _question_index(step: DecodeStep) -> Optional[int]:
    for key in ("question_index", "question_idx", "question_number"):
        value = step.metadata.get(key)
        if value is not None:
            return int(value)
    return None


def _eagle_candidates(step: DecodeStep) -> List[OracleCandidate]:
    return sorted(
        [candidate for candidate in step.candidates if candidate.source == "eagle"],
        key=lambda item: (
            item.tree_index is None,
            item.tree_index if item.tree_index is not None else item.depth,
            item.depth,
            item.token,
        ),
    )


def _sam_candidates(step: DecodeStep) -> List[OracleCandidate]:
    return [candidate for candidate in step.candidates if candidate.source == "sam"]


def _eligible_margin_candidates(step: DecodeStep) -> List[OracleCandidate]:
    return [
        candidate
        for candidate in _eagle_candidates(step)
        if candidate.sibling_margin is not None
    ]


def _candidate_parent_prefix(candidate: OracleCandidate) -> Tuple[int, ...]:
    path = candidate_nonroot_path(candidate)
    return tuple(path[:-1])


def _true_parent_prefix(step: DecodeStep) -> Optional[Tuple[int, ...]]:
    if step.first_rejected_depth is None:
        return None
    return tuple(step.acceptance_path[: max(int(step.first_rejected_depth) - 1, 0)])


def _risk_score(candidate: OracleCandidate) -> Optional[float]:
    if candidate.sibling_margin is None or candidate.local_logprob is None:
        return None
    depth = max(int(candidate.depth), 1)
    if candidate.cumulative_path_logprob is None:
        avg_path_logprob = float(candidate.local_logprob)
    else:
        avg_path_logprob = float(candidate.cumulative_path_logprob) / float(depth)
    return (
        -float(candidate.sibling_margin)
        - float(candidate.local_logprob)
        - avg_path_logprob
    )


def predict_low_margin_node(step: DecodeStep, threshold: float) -> Prediction:
    for candidate in _eligible_margin_candidates(step):
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


def predict_high_margin_node(step: DecodeStep, threshold: float) -> Prediction:
    for candidate in _eligible_margin_candidates(step):
        margin = float(candidate.sibling_margin)
        if margin > float(threshold):
            return Prediction(
                method="high_margin_node",
                candidate=candidate,
                depth=int(candidate.depth),
                score=margin,
                threshold=float(threshold),
            )
    return Prediction("high_margin_node", None, None, threshold=float(threshold))


def predict_risk_score_node(step: DecodeStep, threshold: float) -> Prediction:
    scored: List[Tuple[float, OracleCandidate]] = []
    for candidate in _eagle_candidates(step):
        score = _risk_score(candidate)
        if score is not None:
            scored.append((float(score), candidate))
    if not scored:
        return Prediction("risk_score_node", None, None, threshold=float(threshold))
    score, candidate = max(
        scored,
        key=lambda item: (
            item[0],
            -(item[1].tree_index if item[1].tree_index is not None else item[1].depth),
        ),
    )
    if score > float(threshold):
        return Prediction(
            method="risk_score_node",
            candidate=candidate,
            depth=int(candidate.depth),
            score=score,
            threshold=float(threshold),
        )
    return Prediction("risk_score_node", None, None, threshold=float(threshold))


def predict_depth_only_low_confidence(step: DecodeStep, threshold: float) -> Prediction:
    by_depth: Dict[int, List[float]] = {}
    first_by_depth: Dict[int, OracleCandidate] = {}
    for candidate in _eligible_margin_candidates(step):
        depth = int(candidate.depth)
        by_depth.setdefault(depth, []).append(float(candidate.sibling_margin))
        first_by_depth.setdefault(depth, candidate)
    for depth in sorted(by_depth):
        avg_margin = sum(by_depth[depth]) / len(by_depth[depth])
        if avg_margin < float(threshold):
            candidate = first_by_depth[depth]
            return Prediction(
                method="depth_only_low_confidence",
                candidate=candidate,
                depth=depth,
                score=avg_margin,
                threshold=float(threshold),
            )
    return Prediction("depth_only_low_confidence", None, None, threshold=float(threshold))


def predict_depth_prior(step: DecodeStep, depth: int) -> Prediction:
    for candidate in _eagle_candidates(step):
        if int(candidate.depth) == int(depth):
            return Prediction(
                method="depth_prior",
                candidate=candidate,
                depth=int(depth),
                score=float(depth),
                threshold=float(depth),
            )
    return Prediction("depth_prior", None, None, threshold=float(depth))


def _schema_report(steps: Sequence[DecodeStep]) -> Dict[str, Any]:
    total = len(steps)
    labeled = sum(1 for step in steps if step.has_rejection_boundary_label)
    question_indexed = sum(1 for step in steps if _question_index(step) is not None)
    eagle_candidates = [
        candidate
        for step in steps
        for candidate in _eagle_candidates(step)
    ]
    multi_sibling = [
        candidate
        for candidate in eagle_candidates
        if candidate.sibling_margin is not None
    ]
    report = {
        "steps": total,
        "steps_with_rejection_boundary_label": labeled,
        "steps_with_question_index": question_indexed,
        "eagle_candidates": len(eagle_candidates),
        "eagle_candidates_with_tree_index": sum(
            1 for candidate in eagle_candidates if candidate.tree_index is not None
        ),
        "eagle_candidates_with_parent_index": sum(
            1 for candidate in eagle_candidates if candidate.parent_index is not None
        ),
        "eagle_candidates_with_local_logprob": sum(
            1 for candidate in eagle_candidates if candidate.local_logprob is not None
        ),
        "eagle_candidates_with_sibling_margin": len(multi_sibling),
        "eagle_candidates_with_cumulative_path_logprob": sum(
            1 for candidate in eagle_candidates
            if candidate.cumulative_path_logprob is not None
        ),
    }
    report["schema_ok"] = (
        total > 0
        and labeled == total
        and question_indexed == total
        and len(eagle_candidates) > 0
        and report["eagle_candidates_with_tree_index"] > 0
        and report["eagle_candidates_with_local_logprob"] > 0
        and report["eagle_candidates_with_sibling_margin"] > 0
    )
    return report


def require_calibration_schema(steps: Sequence[DecodeStep]) -> Dict[str, Any]:
    report = _schema_report(steps)
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
        if _question_index(step) is not None
        and int(begin) <= int(_question_index(step)) < int(end)
    ]
    if not selected:
        raise ValueError("no decode steps found for question split [{}, {})".format(begin, end))
    return selected


def _is_depth_hit(step: DecodeStep, prediction: Prediction) -> bool:
    return (
        prediction.depth is not None
        and step.first_rejected_depth is not None
        and int(prediction.depth) == int(step.first_rejected_depth)
    )


def _is_prefix_hit(step: DecodeStep, prediction: Prediction) -> bool:
    if prediction.candidate is None or not _is_depth_hit(step, prediction):
        return False
    true_parent = _true_parent_prefix(step)
    return true_parent is not None and _candidate_parent_prefix(prediction.candidate) == true_parent


def _sam_can_rescue(step: DecodeStep) -> bool:
    if step.first_rejected_depth is None:
        return False
    reject_depth = int(step.first_rejected_depth)
    return any(
        candidate.depth >= reject_depth and candidate_matches_acceptance(candidate, step)
        for candidate in _sam_candidates(step)
    )


def _oracle_rank_key(step: DecodeStep, candidate: OracleCandidate) -> Tuple[int, int, float, int]:
    truth = 1 if candidate_matches_acceptance(candidate, step) else 0
    return (-truth, int(candidate.depth), -float(candidate.score), int(candidate.token))


def _select_oracle_allowed(
    step: DecodeStep,
    candidates: Sequence[OracleCandidate],
    node_budget: Optional[int],
) -> List[OracleCandidate]:
    selected = sorted(candidates, key=lambda candidate: _oracle_rank_key(step, candidate))
    if node_budget is None:
        return selected
    return selected[: max(int(node_budget), 0)]


def _prediction_mat(
    step: DecodeStep,
    prediction: Prediction,
    node_budget: Optional[int],
) -> float:
    allowed = list(select_eagle3_only(step, None))
    if _is_prefix_hit(step, prediction) and _sam_can_rescue(step):
        allowed.extend(_sam_candidates(step))
    selected = _select_oracle_allowed(step, allowed, node_budget)
    return selected_mat(step, selected)


def evaluate_predictions(
    steps: Sequence[DecodeStep],
    predictions: Sequence[Prediction],
    method: str,
    threshold: Optional[float] = None,
    node_budget: Optional[int] = DEFAULT_NODE_BUDGET,
) -> Dict[str, Any]:
    if len(steps) != len(predictions):
        raise ValueError("steps and predictions length mismatch")
    total = len(steps)
    true_rejections = sum(1 for step in steps if step.first_rejected_depth is not None)
    triggered = sum(1 for prediction in predictions if prediction.depth is not None)
    depth_hits = sum(
        1 for step, prediction in zip(steps, predictions)
        if _is_depth_hit(step, prediction)
    )
    prefix_hits = sum(
        1 for step, prediction in zip(steps, predictions)
        if _is_prefix_hit(step, prediction)
    )
    rescued_hits = sum(
        1 for step, prediction in zip(steps, predictions)
        if _is_prefix_hit(step, prediction) and _sam_can_rescue(step)
    )
    depth_errors = [
        abs(int(prediction.depth) - int(step.first_rejected_depth))
        for step, prediction in zip(steps, predictions)
        if prediction.depth is not None and step.first_rejected_depth is not None
    ]
    eagle_mat = sum(
        selected_mat(step, select_eagle3_only(step, node_budget))
        for step in steps
    ) / float(total)
    perfect_mat = sum(
        selected_mat(step, select_perfect(step, node_budget))
        for step in steps
    ) / float(total)
    predicted_mat = sum(
        _prediction_mat(step, prediction, node_budget)
        for step, prediction in zip(steps, predictions)
    ) / float(total)
    base_rate = true_rejections / float(total) if total else 0.0
    precision = prefix_hits / float(triggered) if triggered else 0.0
    recall = prefix_hits / float(true_rejections) if true_rejections else 0.0
    ceiling = perfect_mat - eagle_mat
    return {
        "method": method,
        "threshold": threshold,
        "node_budget": node_budget,
        "steps": total,
        "true_rejection_steps": true_rejections,
        "triggered_steps": triggered,
        "trigger_rate": triggered / float(total) if total else 0.0,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "precision_lift": precision / base_rate if base_rate > 0 else None,
        "depth_hit_rate": depth_hits / float(triggered) if triggered else 0.0,
        "path_prefix_hit_rate": prefix_hits / float(depth_hits) if depth_hits else 0.0,
        "sam_rescue_rate_on_prefix_hits": rescued_hits / float(prefix_hits) if prefix_hits else 0.0,
        "median_depth_error": statistics.median(depth_errors) if depth_errors else None,
        "eagle_mat": eagle_mat,
        "perfect_mat": perfect_mat,
        "predicted_boundary_oracle_mat": predicted_mat,
        "oracle_gap_vs_eagle": (
            (predicted_mat - eagle_mat) / eagle_mat if eagle_mat > 0 else None
        ),
        "recovered_oracle_ceiling": (
            (predicted_mat - eagle_mat) / ceiling if ceiling > 0 else None
        ),
    }


def _threshold_grid(
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


def _choose_best(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("empty sweep")
    eligible = [
        row for row in rows
        if row["trigger_rate"] <= MAX_TRIGGER_RATE
        and row["boundary_precision"] >= MIN_BOUNDARY_PRECISION
    ]
    best_unconstrained = max(
        rows,
        key=lambda row: (
            row["predicted_boundary_oracle_mat"],
            row["boundary_recall"],
            -row["trigger_rate"],
        ),
    )
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["predicted_boundary_oracle_mat"],
                row["boundary_recall"],
                -row["trigger_rate"],
            ),
        )
        if eligible
        else best_unconstrained
    )
    selected = dict(selected)
    selected["selection_constraints_met"] = bool(eligible)
    selected["selected_by"] = "constrained" if eligible else "unconstrained_diagnostic"
    selected["best_unconstrained_threshold"] = best_unconstrained.get("threshold")
    selected["best_unconstrained_trigger_rate"] = best_unconstrained.get("trigger_rate")
    selected["best_unconstrained_oracle_mat"] = best_unconstrained.get(
        "predicted_boundary_oracle_mat"
    )
    return selected


def _run_threshold_sweep(
    steps: Sequence[DecodeStep],
    method: str,
    predictor: Callable[[DecodeStep, float], Prediction],
    thresholds: Sequence[float],
    node_budget: Optional[int],
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
    return rows, _choose_best(rows)


def _fixed_depth_prior(train_steps: Sequence[DecodeStep]) -> int:
    depths = [
        int(step.first_rejected_depth)
        for step in train_steps
        if step.first_rejected_depth is not None
    ]
    if not depths:
        return 1
    return Counter(depths).most_common(1)[0][0]


def _matched_random_predictions(
    steps: Sequence[DecodeStep],
    target_trigger_rate: float,
    seed: int,
) -> List[Prediction]:
    rng = random.Random(int(seed))
    predictions: List[Prediction] = []
    for step in steps:
        candidates = _eagle_candidates(step)
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


def calibrate(
    steps: Sequence[DecodeStep],
    train_begin: int,
    train_end: int,
    valid_begin: int,
    valid_end: int,
    max_thresholds: int = DEFAULT_MAX_SWEEP_THRESHOLDS,
    node_budget: Optional[int] = DEFAULT_NODE_BUDGET,
) -> Dict[str, Any]:
    schema = require_calibration_schema(steps)
    train_steps = split_steps(steps, train_begin, train_end)
    valid_steps = split_steps(steps, valid_begin, valid_end)

    margin_values = [
        float(candidate.sibling_margin)
        for step in train_steps
        for candidate in _eligible_margin_candidates(step)
    ]
    risk_values = [
        float(score)
        for step in train_steps
        for candidate in _eagle_candidates(step)
        for score in [_risk_score(candidate)]
        if score is not None
    ]
    margin_thresholds = _threshold_grid(margin_values, max_thresholds=max_thresholds)
    risk_thresholds = _threshold_grid(risk_values, max_thresholds=max_thresholds)
    if not margin_thresholds:
        raise ValueError("no sibling_margin values available for threshold sweep")
    if not risk_thresholds:
        raise ValueError("no risk-score values available for threshold sweep")

    train_sweeps: Dict[str, List[Dict[str, Any]]] = {}
    selected: Dict[str, Dict[str, Any]] = {}
    for method, predictor, thresholds in (
        ("low_margin_node", predict_low_margin_node, margin_thresholds),
        ("high_margin_node", predict_high_margin_node, margin_thresholds),
        ("depth_only_low_confidence", predict_depth_only_low_confidence, margin_thresholds),
        ("risk_score_node", predict_risk_score_node, risk_thresholds),
    ):
        rows, best = _run_threshold_sweep(
            train_steps,
            method,
            predictor,
            thresholds,
            node_budget=node_budget,
        )
        train_sweeps[method] = rows
        selected[method] = best

    depth_prior = _fixed_depth_prior(train_steps)
    depth_prior_train = evaluate_predictions(
        train_steps,
        [predict_depth_prior(step, depth_prior) for step in train_steps],
        method="depth_prior",
        threshold=float(depth_prior),
        node_budget=node_budget,
    )
    selected["depth_prior"] = depth_prior_train

    low_margin_trigger_rate = selected["low_margin_node"]["trigger_rate"]
    random_rows = []
    random_valid_rows = []
    for seed in range(10):
        random_rows.append(
            evaluate_predictions(
                train_steps,
                _matched_random_predictions(train_steps, low_margin_trigger_rate, seed),
                method="random_node",
                threshold=low_margin_trigger_rate,
                node_budget=node_budget,
            )
        )
        random_valid_rows.append(
            evaluate_predictions(
                valid_steps,
                _matched_random_predictions(valid_steps, low_margin_trigger_rate, seed),
                method="random_node",
                threshold=low_margin_trigger_rate,
                node_budget=node_budget,
            )
        )
    selected["random_node"] = _average_metric_rows(random_rows, "random_node")

    heldout: Dict[str, Dict[str, Any]] = {}
    for method, predictor in (
        ("low_margin_node", predict_low_margin_node),
        ("high_margin_node", predict_high_margin_node),
        ("depth_only_low_confidence", predict_depth_only_low_confidence),
        ("risk_score_node", predict_risk_score_node),
    ):
        threshold = float(selected[method]["threshold"])
        heldout[method] = evaluate_predictions(
            valid_steps,
            [predictor(step, threshold) for step in valid_steps],
            method=method,
            threshold=threshold,
            node_budget=node_budget,
        )
    heldout["depth_prior"] = evaluate_predictions(
        valid_steps,
        [predict_depth_prior(step, depth_prior) for step in valid_steps],
        method="depth_prior",
        threshold=float(depth_prior),
        node_budget=node_budget,
    )
    heldout["random_node"] = _average_metric_rows(random_valid_rows, "random_node")

    gates = {
        "trigger_rate_lte_15pct": heldout["low_margin_node"]["trigger_rate"] <= 0.15,
        "boundary_precision_gte_20pct": heldout["low_margin_node"]["boundary_precision"] >= 0.20,
        "boundary_recall_gte_35pct": heldout["low_margin_node"]["boundary_recall"] >= 0.35,
        "precision_lift_gte_3x": (
            heldout["low_margin_node"]["precision_lift"] is not None
            and heldout["low_margin_node"]["precision_lift"] >= 3.0
        ),
        "median_depth_error_lte_1": (
            heldout["low_margin_node"]["median_depth_error"] is not None
            and heldout["low_margin_node"]["median_depth_error"] <= 1
        ),
        "path_prefix_hit_rate_gte_50pct": heldout["low_margin_node"]["path_prefix_hit_rate"] >= 0.50,
        "oracle_mat_gte_eagle_1p03": (
            heldout["low_margin_node"]["predicted_boundary_oracle_mat"]
            >= heldout["low_margin_node"]["eagle_mat"] * 1.03
        ),
        "recovered_ceiling_gte_30pct": (
            heldout["low_margin_node"]["recovered_oracle_ceiling"] is not None
            and heldout["low_margin_node"]["recovered_oracle_ceiling"] >= 0.30
        ),
        "low_margin_beats_high_margin": (
            heldout["low_margin_node"]["predicted_boundary_oracle_mat"]
            > heldout["high_margin_node"]["predicted_boundary_oracle_mat"]
        ),
        "node_level_beats_depth_only": (
            heldout["low_margin_node"]["predicted_boundary_oracle_mat"]
            > heldout["depth_only_low_confidence"]["predicted_boundary_oracle_mat"]
        ),
    }

    return {
        "schema_report": schema,
        "splits": {
            "train": {"begin": train_begin, "end": train_end, "steps": len(train_steps)},
            "heldout": {"begin": valid_begin, "end": valid_end, "steps": len(valid_steps)},
        },
        "threshold_grid": {
            "max_thresholds": int(max_thresholds),
            "margin_values": len(margin_values),
            "margin_thresholds": len(margin_thresholds),
            "risk_values": len(risk_values),
            "risk_thresholds": len(risk_thresholds),
        },
        "node_budget": node_budget,
        "selected_thresholds": selected,
        "train_sweeps": train_sweeps,
        "heldout_metrics": heldout,
        "decision_gates": gates,
        "pass": all(gates.values()),
    }


def _average_metric_rows(rows: Sequence[Dict[str, Any]], method: str) -> Dict[str, Any]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", action="append", required=True)
    parser.add_argument("--answer-file", default=None)
    parser.add_argument("--train-begin", type=int, required=True)
    parser.add_argument("--train-end", type=int, required=True)
    parser.add_argument("--valid-begin", type=int, required=True)
    parser.add_argument("--valid-end", type=int, required=True)
    parser.add_argument(
        "--max-thresholds",
        type=int,
        default=DEFAULT_MAX_SWEEP_THRESHOLDS,
        help="Maximum thresholds per sweep; bounds CPU time on dense float traces.",
    )
    parser.add_argument(
        "--node-budget",
        type=int,
        default=DEFAULT_NODE_BUDGET,
        help=(
            "Candidate node budget used for eagle/perfect/predicted MAT. "
            "Keep this aligned with oracle_fusion_analysis --top-k."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_paths = [Path(path) for path in args.trace_file]
    for path in trace_paths:
        if not path.exists():
            raise SystemExit("ERROR: trace file not found: {}".format(path))
    if args.answer_file is not None and not Path(args.answer_file).exists():
        warn("answer file not found; split metadata must come from profile steps: {}".format(args.answer_file))
    steps = load_decode_traces(trace_paths)
    if not steps:
        raise SystemExit("ERROR: no decode steps with candidates found")
    print("Loaded decode steps: {}".format(len(steps)), flush=True)
    print(
        "Running calibration sweep with max_thresholds={}, node_budget={}...".format(
            args.max_thresholds,
            args.node_budget,
        ),
        flush=True,
    )
    try:
        result = calibrate(
            steps,
            train_begin=args.train_begin,
            train_end=args.train_end,
            valid_begin=args.valid_begin,
            valid_end=args.valid_end,
            max_thresholds=args.max_thresholds,
            node_budget=args.node_budget,
        )
    except ValueError as exc:
        raise SystemExit("ERROR: {}".format(exc))
    output_dir = Path(args.output_dir)
    write_outputs(result, output_dir)
    low = result["heldout_metrics"]["low_margin_node"]
    selected_low = result["selected_thresholds"]["low_margin_node"]
    print("Boundary predictor calibration")
    print("steps_train:", result["splits"]["train"]["steps"])
    print("steps_heldout:", result["splits"]["heldout"]["steps"])
    print("node_budget:", result["node_budget"])
    print("low_margin_threshold:", selected_low["threshold"])
    print("low_margin_selected_by:", selected_low.get("selected_by", "n/a"))
    print("low_margin_selection_constraints_met:", bool(selected_low.get("selection_constraints_met", True)))
    if not selected_low.get("selection_constraints_met", True):
        print(
            "WARNING: no low-margin threshold met training trigger/precision constraints; "
            "reporting the best unconstrained diagnostic threshold.",
            file=sys.stderr,
        )
    print("margin_thresholds:", result["threshold_grid"]["margin_thresholds"])
    print("risk_thresholds:", result["threshold_grid"]["risk_thresholds"])
    print("heldout_trigger_rate: {:.2%}".format(low["trigger_rate"]))
    print("heldout_boundary_precision: {:.2%}".format(low["boundary_precision"]))
    print("heldout_boundary_recall: {:.2%}".format(low["boundary_recall"]))
    print("heldout_predicted_boundary_oracle_mat: {:.4f}".format(low["predicted_boundary_oracle_mat"]))
    print("pass:", bool(result["pass"]))
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
