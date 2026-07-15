#!/usr/bin/env python3
"""Compatibility shim for ``evaluation.oracle.depth_decoupled``."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.oracle.depth_decoupled import *  # noqa: F401,F403
from evaluation.oracle.depth_decoupled import main


if __name__ == "__main__":
    main()
