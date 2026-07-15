"""Boundary-predictor scaffolding for offline calibration."""

from evaluation.boundary.calibrate import (
    calibrate,
    require_calibration_schema,
    schema_report,
    split_steps,
    write_outputs,
)
from evaluation.boundary.candidates import eagle_candidates, sam_candidates
from evaluation.boundary.low_margin import predict_low_margin_node
from evaluation.boundary.metrics import evaluate_predictions, is_prefix_hit, sam_can_rescue
from evaluation.boundary.selection import choose_best, constrain_selection, selection_rank
from evaluation.boundary.sweep import run_threshold_sweep, threshold_grid
from evaluation.boundary.types import (
    DEFAULT_MAX_SWEEP_THRESHOLDS,
    DEFAULT_NODE_BUDGET,
    MAX_TRIGGER_RATE,
    MIN_BOUNDARY_PRECISION,
    Prediction,
    validate_probability_threshold,
)

__all__ = [
    "DEFAULT_MAX_SWEEP_THRESHOLDS",
    "DEFAULT_NODE_BUDGET",
    "MAX_TRIGGER_RATE",
    "MIN_BOUNDARY_PRECISION",
    "Prediction",
    "calibrate",
    "choose_best",
    "constrain_selection",
    "eagle_candidates",
    "evaluate_predictions",
    "is_prefix_hit",
    "predict_low_margin_node",
    "require_calibration_schema",
    "run_threshold_sweep",
    "sam_can_rescue",
    "sam_candidates",
    "schema_report",
    "selection_rank",
    "split_steps",
    "threshold_grid",
    "validate_probability_threshold",
    "write_outputs",
]
