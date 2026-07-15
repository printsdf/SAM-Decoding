#!/usr/bin/env python3
"""Compatibility shim for ``evaluation.boundary``.

Prefer::

    python -m evaluation.boundary.cli
    from evaluation.boundary import calibrate, evaluate_predictions, ...
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.boundary import (  # noqa: F401
    DEFAULT_MAX_SWEEP_THRESHOLDS,
    DEFAULT_NODE_BUDGET,
    MAX_TRIGGER_RATE,
    MIN_BOUNDARY_PRECISION,
    Prediction,
    calibrate,
    choose_best,
    constrain_selection,
    eagle_candidates,
    evaluate_predictions,
    is_prefix_hit,
    predict_low_margin_node,
    require_calibration_schema,
    run_threshold_sweep,
    sam_can_rescue,
    sam_candidates,
    schema_report,
    selection_rank,
    split_steps,
    threshold_grid,
    validate_probability_threshold,
    write_outputs,
)
from evaluation.boundary.calibrate import (  # noqa: F401
    average_metric_rows,
    hard_fail_heldout,
    matched_random_predictions,
    warn,
)
from evaluation.boundary.candidates import (  # noqa: F401
    candidate_parent_prefix,
    eligible_margin_candidates,
    question_index,
    true_parent_prefix,
)
from evaluation.boundary.cli import main, parse_args  # noqa: F401
from evaluation.boundary.metrics import (  # noqa: F401
    is_depth_hit,
    oracle_rank_key,
    prediction_mat,
    select_oracle_allowed,
)
from evaluation.boundary.selection import choose_best as _choose_best  # noqa: F401
from evaluation.boundary.sweep import (  # noqa: F401
    run_threshold_sweep as _run_threshold_sweep,
    threshold_grid as _threshold_grid,
)
from evaluation.boundary.candidates import eagle_candidates as _eagle_candidates  # noqa: F401
from evaluation.boundary.metrics import (  # noqa: F401
    is_prefix_hit as _is_prefix_hit,
    sam_can_rescue as _sam_can_rescue,
)


if __name__ == "__main__":
    main()
