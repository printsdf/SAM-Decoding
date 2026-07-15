"""CLI for multi-method same-trace oracle analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from evaluation.oracle.analyze import analyze_steps, render_table
from evaluation.oracle.load import load_decode_traces, warn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline oracle analysis for EAGLE3 + SAM fusion traces."
    )
    parser.add_argument("trace_files", nargs="*", help="Oracle trace JSON/JSONL files.")
    parser.add_argument("--trace-file", action="append", default=[], help="Oracle trace JSON/JSONL file.")
    parser.add_argument("--top-k", type=int, default=60, help="Candidate budget for perfect/source/baseline selectors.")
    parser.add_argument("--node-budget", type=int, default=None, help="Node budget for the budgeted selector. Defaults to --top-k.")
    parser.add_argument("--depth-max-split", type=int, default=10, help="Maximum D_split for the depth_decoupled selector.")
    parser.add_argument(
        "--gap-baseline",
        default="eagle3_only",
        choices=["eagle3_only", "sam_sequence_graft"],
        help="Baseline MAT used in (oracle_MAT - baseline_MAT) / baseline_MAT.",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in list(args.trace_file) + list(args.trace_files)]
    if not paths:
        raise SystemExit("ERROR: provide at least one trace file")
    steps = load_decode_traces(paths)
    if not steps:
        raise SystemExit("ERROR: no decode steps with candidates found")
    results = analyze_steps(
        steps,
        top_k=args.top_k,
        node_budget=args.node_budget,
        gap_baseline=args.gap_baseline,
        depth_max_split=args.depth_max_split,
    )
    print(render_table(results, args.gap_baseline))
    if args.output_json:
        depth_analysis: Dict[str, Any] = {}
        try:
            from evaluation.oracle.depth_decoupled import DepthStratifiedOracle

            depth_analysis = DepthStratifiedOracle(steps).analyze(
                max_split=args.depth_max_split
            )
        except Exception as exc:
            warn("depth_decoupled JSON payload skipped: {}".format(exc))
        output = {
            "schema_version": 1,
            "trace_files": [str(path) for path in paths],
            "steps": len(steps),
            "top_k": args.top_k,
            "node_budget": args.node_budget,
            "gap_baseline": args.gap_baseline,
            "methods": [row.to_dict() for row in results],
            "results": [row.to_dict() for row in results],
        }
        output.update(depth_analysis)
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
