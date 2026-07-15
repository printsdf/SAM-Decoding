#!/usr/bin/env python3
"""Compatibility shim for ``evaluation.oracle.rejection_boundary``."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.oracle.rejection_boundary import (  # noqa: F401
    compute_rescue_stats,
    find_first_rejection_depth,
    main,
    sam_can_rescue,
)


if __name__ == "__main__":
    main()
