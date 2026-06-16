#!/usr/bin/env python3
"""Oracle analysis for high-precision SAM filtering.

Measures oracle gap when filtering SAM candidates by match quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.oracle_fusion_analysis import (
    DecodeStep,
    candidate_matches_acceptance,
    coerce_decode_steps,
    load_decode_traces,
)


def compute_oracle_mat_with_filter(
    steps: List[DecodeStep],
    min_match_length: int
) -> Dict[str, Any]:
    """Compute oracle MAT with SAM filtered by match quality (using depth as proxy)."""
    stats = {
        "threshold": min_match_length,
        "total_steps": len(steps),
        "steps_with_sam": 0,
        "steps_with_filtered_sam": 0,
        "avg_sam_count": 0.0,
        "avg_filtered_sam_count": 0.0,
        "baseline_mat": 0.0,
        "oracle_mat": 0.0,
        "oracle_gap": 0.0,
    }

    baseline_tokens = 0
    oracle_tokens = 0
    sam_counts = []
    filtered_sam_counts = []

    for step in steps:
        eagle_candidates = [c for c in step.candidates if c.source == "eagle"]
        sam_candidates = [c for c in step.candidates if c.source == "sam"]

        # Filter SAM by depth (proxy for match_length since actual field may not exist)
        # Longer SAM paths indicate better matches from the cache
        filtered_sam = [
            c for c in sam_candidates
            if c.depth >= min_match_length
        ]

        sam_counts.append(len(sam_candidates))
        filtered_sam_counts.append(len(filtered_sam))

        if sam_candidates:
            stats["steps_with_sam"] += 1
        if filtered_sam:
            stats["steps_with_filtered_sam"] += 1

        # Baseline: eagle-only MAT
        eagle_accepted = [
            c for c in eagle_candidates
            if candidate_matches_acceptance(c, step)
        ]
        baseline_depth = max((c.depth for c in eagle_accepted), default=0)
        baseline_tokens += baseline_depth

        # Oracle: eagle + filtered_sam
        combined_candidates = eagle_candidates + filtered_sam
        oracle_accepted = [
            c for c in combined_candidates
            if candidate_matches_acceptance(c, step)
        ]
        oracle_depth = max((c.depth for c in oracle_accepted), default=0)
        oracle_tokens += oracle_depth

    stats["avg_sam_count"] = sum(sam_counts) / len(sam_counts) if sam_counts else 0.0
    stats["avg_filtered_sam_count"] = sum(filtered_sam_counts) / len(filtered_sam_counts) if filtered_sam_counts else 0.0
    stats["coverage"] = stats["steps_with_filtered_sam"] / stats["total_steps"] if stats["total_steps"] > 0 else 0.0
    stats["baseline_mat"] = baseline_tokens / stats["total_steps"] if stats["total_steps"] > 0 else 0.0
    stats["oracle_mat"] = oracle_tokens / stats["total_steps"] if stats["total_steps"] > 0 else 0.0
    stats["oracle_gap"] = (
        (stats["oracle_mat"] - stats["baseline_mat"]) / stats["baseline_mat"]
        if stats["baseline_mat"] > 0 else 0.0
    )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Oracle analysis for high-precision SAM filtering"
    )
    parser.add_argument(
        "--trace-file",
        required=True,
        help="Fusion profile JSON/JSONL trace file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="JSON output path"
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="5,8,10,12,15,20",
        help="Comma-separated match_length thresholds to test (default: 5,8,10,12,15,20)"
    )

    args = parser.parse_args()

    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"ERROR: Trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    thresholds = [int(x.strip()) for x in args.thresholds.split(",")]

    # Load trace
    print(f"Loading trace: {trace_path}")
    steps = coerce_decode_steps(load_decode_traces([trace_path]))
    print(f"Loaded {len(steps)} decode steps")

    # Compute stats for each threshold
    print("\nAnalyzing high-precision SAM filtering...")
    results = []

    for threshold in thresholds:
        print(f"  Testing threshold: match_length >= {threshold}")
        stats = compute_oracle_mat_with_filter(steps, threshold)
        results.append(stats)

    # Find optimal threshold
    optimal = max(results, key=lambda x: x["oracle_gap"])

    # Print summary
    print("\n" + "=" * 80)
    print("High-Precision SAM Oracle Analysis")
    print("=" * 80)
    print(f"{'Threshold':<12} {'SAM Count':<12} {'Coverage':<12} {'Oracle MAT':<12} {'Gap':<12}")
    print("-" * 80)

    baseline_mat = results[0]["baseline_mat"]
    print(f"{'baseline':<12} {'-':<12} {'-':<12} {baseline_mat:<12.4f} {'+0.00%':<12}")

    for r in results:
        print(
            f"{r['threshold']:<12} "
            f"{r['avg_filtered_sam_count']:<12.2f} "
            f"{r['coverage']:<12.1%} "
            f"{r['oracle_mat']:<12.4f} "
            f"{r['oracle_gap']:<+11.2%} "
        )

    print()
    print(f"Optimal threshold: {optimal['threshold']} (gap: {optimal['oracle_gap']:+.2%})")
    print()

    # Save results
    output_data = {
        "baseline_mat": baseline_mat,
        "results_by_threshold": results,
        "optimal_threshold": optimal["threshold"],
        "max_gap": optimal["oracle_gap"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
