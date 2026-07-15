"""Schema and raw-logit diagnostics for Drafter-MARS."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from evaluation.boundary import eagle_candidates
from evaluation.drafter_mars.parents import ParentRecord, eagle_parent_records
from evaluation.oracle import DecodeStep

DRAFT_MARS_THETA_GRID = (
    0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98,
)
DRAFT_MARS_REACHABLE_QUANTILES = (0.50, 0.70, 0.85, 0.95)


def raw_logit_diagnostics(steps: Sequence[DecodeStep]) -> Dict[str, Any]:
    """Report ratio instability from non-positive drafter logits."""
    parents: List[ParentRecord] = []
    for step in steps:
        parents.extend(eagle_parent_records(step))
    parents = [p for p in parents if p.z1 is not None and p.z2 is not None]
    total = len(parents)
    top1_nonpositive = sum(1 for p in parents if float(p.z1) <= 0.0)
    both_negative = sum(
        1 for p in parents if float(p.z1) < 0.0 and float(p.z2) < 0.0
    )
    return {
        "raw_logit_parent_count": total,
        "top1_nonpositive_rate": (top1_nonpositive / total if total else 0.0),
        "both_negative_rate": (both_negative / total if total else 0.0),
    }


def schema_report(steps: Sequence[DecodeStep]) -> Dict[str, Any]:
    """Report whether the trace carries the Drafter-MARS schema fields."""
    candidates = [
        candidate
        for step in steps
        for candidate in eagle_candidates(step)
    ]
    total = len(candidates)
    with_raw = sum(
        1
        for c in candidates
        if c.parent_top1_logit is not None and c.parent_top2_logit is not None
    )
    with_logprob = sum(1 for c in candidates if c.local_logprob is not None)
    with_cumulative = sum(
        1 for c in candidates if c.cumulative_path_logprob is not None
    )
    report = {
        "eagle_candidates": total,
        "eagle_candidates_with_parent_raw_logits": with_raw,
        "eagle_candidates_with_local_logprob": with_logprob,
        "eagle_candidates_with_cumulative_path_logprob": with_cumulative,
        "theta_grid": list(DRAFT_MARS_THETA_GRID),
        "reachable_quantiles": list(DRAFT_MARS_REACHABLE_QUANTILES),
    }
    report["raw_logits_available"] = total > 0 and with_raw == total
    report["schema_ok"] = bool(report["raw_logits_available"])
    report.update(raw_logit_diagnostics(steps))
    return report


def require_drafter_mars_schema(steps: Sequence[DecodeStep]) -> Dict[str, Any]:
    report = schema_report(steps)
    if not report["schema_ok"]:
        raise ValueError(
            "trace is missing Drafter-MARS parent raw-logit fields: {}".format(
                {
                    "eagle_candidates": report["eagle_candidates"],
                    "with_parent_raw_logits": report["eagle_candidates_with_parent_raw_logits"],
                }
            )
        )
    return report
