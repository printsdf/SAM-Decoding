"""Oracle analysis for rejection-boundary SAM rescue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.oracle.load import coerce_decode_steps, load_decode_traces
from evaluation.oracle.select import (
    candidate_matches_acceptance,
    select_eagle3_only,
    selected_mat,
)
from evaluation.oracle.types import DecodeStep


def find_first_rejection_depth(step: DecodeStep, source: str = "eagle") -> Optional[int]:
    """Find the first depth where source candidates fail to match acceptance."""
    source_candidates = [c for c in step.candidates if c.source == source]
    if not source_candidates:
        return None

    accepted_path = step.acceptance_path
    for depth in range(1, len(accepted_path) + 1):
        matches = any(
            c.depth == depth and candidate_matches_acceptance(c, step)
            for c in source_candidates
        )
        if not matches:
            return depth
    return None


def sam_can_rescue(step: DecodeStep, reject_depth: Optional[int]) -> bool:
    """Check if SAM has candidates that extend accepted path from reject_depth."""
    if reject_depth is None or reject_depth <= 0:
        return False

    sam_candidates = [c for c in step.candidates if c.source == "sam"]
    for c in sam_candidates:
        if c.depth < reject_depth:
            continue
        if candidate_matches_acceptance(c, step):
            return True
    return False


def compute_rescue_stats(steps: List[DecodeStep]) -> Dict[str, Any]:
    """Compute rejection-boundary rescue statistics."""
    stats = {
        "total_steps": len(steps),
        "eagle_rejection_steps": 0,
        "sam_rescue_success": 0,
        "sam_rescue_fail": 0,
        "rejection_depth_dist": {},
        "baseline_mat": 0.0,
        "rescue_added_tokens": 0,
    }

    baseline_mats = []

    for step in steps:
        eagle_mat = selected_mat(step, select_eagle3_only(step, None))
        baseline_mats.append(eagle_mat)

        reject_depth = find_first_rejection_depth(step, "eagle")

        if reject_depth is not None:
            stats["eagle_rejection_steps"] += 1

            rd_key = str(reject_depth)
            stats["rejection_depth_dist"][rd_key] = stats["rejection_depth_dist"].get(rd_key, 0) + 1

            if sam_can_rescue(step, reject_depth):
                stats["sam_rescue_success"] += 1
                sam_candidates = [c for c in step.candidates if c.source == "sam"]
                sam_accepted = [c for c in sam_candidates if candidate_matches_acceptance(c, step)]
                sam_mat = selected_mat(step, sam_accepted)
                added = max(0, sam_mat - eagle_mat)
                stats["rescue_added_tokens"] += added
            else:
                stats["sam_rescue_fail"] += 1

    stats["baseline_mat"] = sum(baseline_mats) / len(baseline_mats) if baseline_mats else 0.0

    if stats["eagle_rejection_steps"] > 0:
        stats["sam_rescue_rate"] = stats["sam_rescue_success"] / stats["eagle_rejection_steps"]
    else:
        stats["sam_rescue_rate"] = 0.0

    avg_rescue_gain = (
        stats["rescue_added_tokens"] / stats["sam_rescue_success"]
        if stats["sam_rescue_success"] > 0
        else 0.0
    )
    rescue_coverage = (
        stats["eagle_rejection_steps"] / stats["total_steps"]
        if stats["total_steps"] > 0
        else 0.0
    )
    oracle_mat_gain = avg_rescue_gain * rescue_coverage * stats["sam_rescue_rate"]
    stats["oracle_mat_with_rescue"] = stats["baseline_mat"] + oracle_mat_gain
    stats["oracle_gap"] = (
        oracle_mat_gain / stats["baseline_mat"] if stats["baseline_mat"] > 0 else 0.0
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle analysis for rejection-boundary SAM rescue"
    )
    parser.add_argument(
        "--trace-file",
        required=True,
        help="Fusion profile JSON/JSONL trace file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSON output path",
    )

    args = parser.parse_args()

    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print("ERROR: Trace file not found: {}".format(trace_path), file=sys.stderr)
        sys.exit(1)

    print("Loading trace: {}".format(trace_path))
    steps = coerce_decode_steps(load_decode_traces([trace_path]))
    print("Loaded {} decode steps".format(len(steps)))

    print("Analyzing rejection boundaries...")
    stats = compute_rescue_stats(steps)

    print("\n" + "=" * 60)
    print("Rejection-Boundary SAM Rescue Analysis")
    print("=" * 60)
    print("Total steps: {}".format(stats["total_steps"]))
    total = stats["total_steps"] or 1
    print(
        "EAGLE rejection steps: {} ({:.1%})".format(
            stats["eagle_rejection_steps"],
            stats["eagle_rejection_steps"] / total,
        )
    )
    print("SAM rescue success: {}".format(stats["sam_rescue_success"]))
    print("SAM rescue fail: {}".format(stats["sam_rescue_fail"]))
    print("SAM rescue rate: {:.1%}".format(stats["sam_rescue_rate"]))
    print("\nBaseline MAT (eagle-only): {:.4f}".format(stats["baseline_mat"]))
    print("Oracle MAT (with rescue): {:.4f}".format(stats["oracle_mat_with_rescue"]))
    print("Oracle gap: {:.2%}".format(stats["oracle_gap"]))
    print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("Results saved to: {}".format(output_path))


if __name__ == "__main__":
    main()
