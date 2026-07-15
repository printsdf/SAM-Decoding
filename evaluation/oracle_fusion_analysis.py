#!/usr/bin/env python3
"""Compatibility shim for ``evaluation.oracle``.

Prefer::

    python -m evaluation.oracle.cli
    from evaluation.oracle import load_decode_traces, select_eagle3_only, ...
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.oracle import (  # noqa: F401
    DecodeStep,
    MethodResult,
    OracleCandidate,
    analyze_steps,
    candidate_matches_acceptance,
    candidate_nonroot_path,
    coerce_decode_steps,
    load_decode_traces,
    render_table,
    select_budgeted,
    select_eagle3_only,
    select_perfect,
    select_sam_sequence_graft,
    select_source_balanced,
    selected_mat,
    warn,
)
from evaluation.oracle.cli import main, parse_args  # noqa: F401
from evaluation.oracle.select import default_budget as _default_budget  # noqa: F401


if __name__ == "__main__":
    main()
