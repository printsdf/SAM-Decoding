#!/usr/bin/env python3
"""Compatibility shim for ``evaluation.drafter_mars``.

Prefer::

    from evaluation.drafter_mars import calibrate_drafter_mars, ...
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.drafter_mars import (  # noqa: F401
    DRAFT_MARS_REACHABLE_QUANTILES,
    DRAFT_MARS_THETA_GRID,
    ParentRecord,
    calibrate_drafter_mars,
    eagle_parent_records,
    predict_draft_delta_top_path,
    predict_draft_mars_reachable,
    predict_draft_mars_top_path,
    raw_logit_diagnostics,
    require_drafter_mars_schema,
    schema_report,
)

# Private-name aliases used by older tests/imports.
from evaluation.drafter_mars.parents import eagle_parent_records as _eagle_parent_records  # noqa: F401
from evaluation.drafter_mars.predictors import (  # noqa: F401
    earliest_parent as _earliest_parent,
    prediction_from_parent as _prediction_from_parent,
    reachable_threshold as _reachable_threshold,
)
