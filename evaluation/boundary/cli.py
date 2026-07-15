"""CLI for boundary-predictor calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from evaluation.boundary.calibrate import calibrate, warn, write_outputs
from evaluation.boundary.types import DEFAULT_MAX_SWEEP_THRESHOLDS, DEFAULT_NODE_BUDGET
from evaluation.oracle import load_decode_traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline calibration for EAGLE rejection-boundary predictors."
    )
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
            "Keep this aligned with oracle analysis --top-k."
        ),
    )
    parser.add_argument(
        "--predictor-suite",
        choices=("default", "drafter_mars", "all"),
        default="default",
        help="Predictor suite to run: low_margin baseline only, Drafter-MARS only, or both.",
    )
    parser.add_argument(
        "--theta-grid",
        default=None,
        help="Comma-separated Drafter-MARS theta grid (e.g. 0.84,0.86,...).",
    )
    parser.add_argument(
        "--reachable-quantiles",
        default=None,
        help="Comma-separated reachable-quantile grid for draft_mars_reachable.",
    )
    parser.add_argument(
        "--require-raw-draft-logits",
        action="store_true",
        help="Fail fast if the trace lacks Drafter-MARS parent raw-logit fields.",
    )
    parser.add_argument(
        "--stage",
        choices=("smoke", "full"),
        default="full",
        help=(
            "Selection bar stage: smoke (trigger<=20%%, precision>=15%%) or "
            "full (trigger<=15%%, precision>=20%%)."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _parse_float_csv(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def main() -> None:
    args = parse_args()
    trace_paths = [Path(path) for path in args.trace_file]
    for path in trace_paths:
        if not path.exists():
            raise SystemExit("ERROR: trace file not found: {}".format(path))
    if args.answer_file is not None and not Path(args.answer_file).exists():
        warn(
            "answer file not found; split metadata must come from profile steps: {}".format(
                args.answer_file
            )
        )
    steps = load_decode_traces(trace_paths)
    if not steps:
        raise SystemExit("ERROR: no decode steps with candidates found")
    print("Loaded decode steps: {}".format(len(steps)), flush=True)
    print(
        "Running calibration sweep with max_thresholds={}, node_budget={}, "
        "predictor_suite={}, stage={}...".format(
            args.max_thresholds,
            args.node_budget,
            args.predictor_suite,
            args.stage,
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
            predictor_suite=args.predictor_suite,
            theta_grid=_parse_float_csv(args.theta_grid),
            reachable_quantiles=_parse_float_csv(args.reachable_quantiles),
            require_raw_draft_logits=args.require_raw_draft_logits,
            stage=args.stage,
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
    print("low_margin_threshold:", selected_low.get("threshold"))
    print("low_margin_selected_by:", selected_low.get("selected_by", "n/a"))
    print(
        "low_margin_selection_constraints_met:",
        bool(selected_low.get("selection_constraints_met", True)),
    )
    if not selected_low.get("selection_constraints_met", True):
        print(
            "WARNING: no low-margin threshold met the stage's selection bar; "
            "no threshold selected and heldout is a hard fail.",
            file=sys.stderr,
        )
    print("margin_thresholds:", result["threshold_grid"]["margin_thresholds"])
    print("heldout_trigger_rate: {:.2%}".format(low["trigger_rate"]))
    print("heldout_boundary_precision: {:.2%}".format(low["boundary_precision"]))
    print("heldout_boundary_recall: {:.2%}".format(low["boundary_recall"]))
    print("heldout_eagle_mat: {:.4f}".format(low["eagle_mat"]))
    print("heldout_perfect_mat: {:.4f}".format(low["perfect_mat"]))
    print(
        "heldout_predicted_boundary_oracle_mat: {:.4f}".format(
            low["predicted_boundary_oracle_mat"]
        )
    )
    if low["predicted_boundary_oracle_mat"] > low["perfect_mat"] + 1e-9:
        print(
            "WARNING: predicted boundary MAT exceeds same-split perfect MAT; "
            "check node-budget accounting before interpreting this run.",
            file=sys.stderr,
        )
    if "drafter_mars" in result:
        mars = result["drafter_mars"]
        print("Drafter-MARS schema_ok:", mars["schema_report"].get("schema_ok"))
        print(
            "Drafter-MARS top1_nonpositive_rate: {:.2%}".format(
                mars["schema_report"].get("top1_nonpositive_rate", 0.0)
            )
        )
        for method in ("draft_mars_top_path", "draft_delta_top_path", "draft_mars_reachable"):
            sel = mars["selected_thresholds"].get(method, {})
            held = mars["heldout_metrics"].get(method, {})
            print(
                "{}: selected_by={}, threshold={}, trigger_rate={:.2%}, mat={:.4f}".format(
                    method,
                    sel.get("selected_by"),
                    sel.get("threshold"),
                    held.get("trigger_rate", 0.0),
                    held.get("predicted_boundary_oracle_mat", 0.0),
                )
            )
    print("pass:", bool(result["pass"]))
    print("output_dir:", output_dir)


if __name__ == "__main__":
    main()
