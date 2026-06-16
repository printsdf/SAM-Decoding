#!/usr/bin/env python3
"""Oracle analysis for leaf-only SAM tail extension.

This module measures a depth-decoupled upper bound: EAGLE owns the root and
prefix depths up to ``D_split``; SAM is only allowed to extend deeper accepted
EAGLE leaves. Existing fusion profile traces record root-anchored raw SAM
candidates, not SAM regenerated from every EAGLE leaf, so this oracle only
credits recorded SAM candidates whose paths already match the verifier's
accepted suffix beyond the split.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.oracle_fusion_analysis import (
    DecodeStep,
    candidate_matches_acceptance,
    coerce_decode_steps,
    load_decode_traces,
)


DEFAULT_MAX_SPLIT = 10


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom else 0.0


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _result_key(d_split: int) -> str:
    return "depth_decoupled_d{}".format(int(d_split))


def _is_step_like(value: Any) -> bool:
    return (
        hasattr(value, "candidates")
        and hasattr(value, "acceptance_path")
        and hasattr(value, "stats")
    )


class DepthStratifiedOracle:
    """Depth-stratified oracle for EAGLE-prefix, SAM-tail decoding traces."""

    def __init__(self, trace: Any) -> None:
        if _is_step_like(trace):
            self.steps = [trace]
        elif isinstance(trace, (list, tuple)) and all(_is_step_like(item) for item in trace):
            self.steps = list(trace)
        else:
            self.steps = coerce_decode_steps(trace)

    @classmethod
    def from_files(cls, paths: Sequence[Path]) -> "DepthStratifiedOracle":
        return cls(load_decode_traces(paths))

    def _source_truth_depths(self, step: DecodeStep, source: str) -> set:
        return {
            int(candidate.depth)
            for candidate in step.candidates
            if candidate.source == source and candidate_matches_acceptance(candidate, step)
        }

    def _source_accept_depth(self, step: DecodeStep, source: str) -> int:
        accepted_depths = self._source_truth_depths(step, source)
        depth = 0
        for expected_depth in range(1, len(step.acceptance_path) + 1):
            if expected_depth not in accepted_depths:
                break
            depth = expected_depth
        return depth

    def _sam_extension_depth(self, step: DecodeStep, d_split: int, eagle_depth: int) -> int:
        d_split = max(int(d_split), 0)
        if eagle_depth < d_split:
            return 0
        sam_depths = self._source_truth_depths(step, "sam")
        extension = 0
        for depth in range(d_split + 1, len(step.acceptance_path) + 1):
            if depth not in sam_depths:
                break
            extension = depth - d_split
        return extension

    def _node_counts_for_split(self, step: DecodeStep, d_split: int) -> Tuple[int, int]:
        eagle_nodes = sum(
            1
            for candidate in step.candidates
            if candidate.source == "eagle" and int(candidate.depth) <= d_split
        )
        sam_nodes = sum(
            1
            for candidate in step.candidates
            if candidate.source == "sam" and int(candidate.depth) > d_split
        )
        return eagle_nodes, sam_nodes

    def eagle_only_result(self) -> Dict[str, Any]:
        if not self.steps:
            return {
                "method": "eagle_only",
                "mat": 0.0,
                "accepted_nonroot": 0.0,
                "gap": 0.0,
                "eagle_nodes": 0.0,
                "sam_nodes": 0.0,
                "nodes": 0.0,
                "steps": 0,
            }

        accepted_depths = [self._source_accept_depth(step, "eagle") for step in self.steps]
        eagle_nodes = [
            sum(1 for candidate in step.candidates if candidate.source == "eagle")
            for step in self.steps
        ]
        mat = 1.0 + _mean(accepted_depths)
        nodes = _mean(eagle_nodes)
        return {
            "method": "eagle_only",
            "mat": mat,
            "accepted_nonroot": _mean(accepted_depths),
            "gap": 0.0,
            "eagle_nodes": nodes,
            "sam_nodes": 0.0,
            "nodes": nodes,
            "mat_per_node": _safe_div(mat, nodes),
            "steps": len(self.steps),
        }

    def simulate_split(self, d_split: int) -> Dict[str, Any]:
        d_split = int(d_split)
        if d_split < 0:
            raise ValueError("d_split must be non-negative")
        if not self.steps:
            return {
                "method": _result_key(d_split),
                "d_split": d_split,
                "mat": 0.0,
                "gap": None,
                "eagle_nodes": 0.0,
                "sam_nodes": 0.0,
                "nodes": 0.0,
                "steps": 0,
            }

        eagle_only = self.eagle_only_result()
        eagle_only_mat = float(eagle_only["mat"])
        accepted_nonroot: List[float] = []
        eagle_prefix_depths: List[float] = []
        sam_extension_depths: List[float] = []
        eagle_node_counts: List[float] = []
        sam_node_counts: List[float] = []
        active_sam_node_counts: List[float] = []
        prefix_successes = 0
        extension_opportunities = 0
        possible_extension_tokens = 0

        for step in self.steps:
            eagle_depth = self._source_accept_depth(step, "eagle")
            eagle_prefix_depth = min(eagle_depth, d_split)
            sam_extension_depth = self._sam_extension_depth(step, d_split, eagle_depth)
            eagle_nodes, sam_nodes = self._node_counts_for_split(step, d_split)

            if eagle_depth >= d_split:
                prefix_successes += 1
                active_sam_node_counts.append(float(sam_nodes))
                possible_extension_tokens += max(len(step.acceptance_path) - d_split, 0)
                if sam_nodes > 0:
                    extension_opportunities += 1
            else:
                active_sam_node_counts.append(0.0)

            accepted_nonroot.append(float(eagle_prefix_depth + sam_extension_depth))
            eagle_prefix_depths.append(float(eagle_prefix_depth))
            sam_extension_depths.append(float(sam_extension_depth))
            eagle_node_counts.append(float(eagle_nodes))
            sam_node_counts.append(float(sam_nodes))

        mat = 1.0 + _mean(accepted_nonroot)
        eagle_nodes = _mean(eagle_node_counts)
        sam_nodes = _mean(sam_node_counts)
        nodes = eagle_nodes + sam_nodes
        sam_extension_tokens = sum(sam_extension_depths)
        return {
            "method": _result_key(d_split),
            "d_split": d_split,
            "mat": mat,
            "accepted_nonroot": _mean(accepted_nonroot),
            "gap": _safe_div(mat - eagle_only_mat, eagle_only_mat) if eagle_only_mat else None,
            "conditional_oracle_gap": _safe_div(mat - eagle_only_mat, eagle_only_mat)
            if eagle_only_mat
            else None,
            "eagle_nodes": eagle_nodes,
            "sam_nodes": sam_nodes,
            "nodes": nodes,
            "mat_per_node": _safe_div(mat, nodes),
            "eagle_prefix_depth": _mean(eagle_prefix_depths),
            "sam_extension_depth": _mean(sam_extension_depths),
            "sam_extension_tokens": sam_extension_tokens,
            "prefix_success_rate": _safe_div(prefix_successes, len(self.steps)),
            "sam_extension_opportunity_rate": _safe_div(extension_opportunities, len(self.steps)),
            "sam_extension_accept_rate": _safe_div(sam_extension_tokens, possible_extension_tokens),
            "active_sam_nodes": _mean(active_sam_node_counts),
            "eagle_node_share": _safe_div(eagle_nodes, nodes),
            "sam_node_share": _safe_div(sam_nodes, nodes),
            "steps": len(self.steps),
        }

    def sweep(self, max_split: int = DEFAULT_MAX_SPLIT) -> Dict[int, Dict[str, Any]]:
        max_split = max(int(max_split), 0)
        return {
            d_split: self.simulate_split(d_split)
            for d_split in range(1, max_split + 1)
        }

    def find_optimal_split(self, max_split: int = DEFAULT_MAX_SPLIT) -> int:
        results = self.sweep(max_split)
        if not results:
            return 0
        best_split, _ = max(
            results.items(),
            key=lambda item: (
                float(item[1].get("mat", 0.0)),
                float(item[1].get("gap") or 0.0),
                -int(item[0]),
            ),
        )
        return int(best_split)

    def depth_details(self) -> Dict[str, Dict[str, Any]]:
        buckets: Dict[int, Dict[str, float]] = {}
        for step in self.steps:
            accepted_depths = set(range(1, len(step.acceptance_path) + 1))
            for depth in accepted_depths:
                buckets.setdefault(depth, {"steps_with_target": 0.0})
            for depth in accepted_depths:
                buckets[depth]["steps_with_target"] = buckets[depth].get("steps_with_target", 0.0) + 1.0

            for candidate in step.candidates:
                depth = int(candidate.depth)
                source = candidate.source
                bucket = buckets.setdefault(depth, {})
                bucket[source + "_nodes"] = bucket.get(source + "_nodes", 0.0) + 1.0
                bucket[source + "_score_sum"] = bucket.get(source + "_score_sum", 0.0) + float(candidate.score)
                if candidate_matches_acceptance(candidate, step):
                    bucket[source + "_matches"] = bucket.get(source + "_matches", 0.0) + 1.0

        strata: Dict[str, Dict[str, Any]] = {}
        steps = len(self.steps)
        for depth in sorted(buckets):
            bucket = buckets[depth]
            eagle_nodes = bucket.get("eagle_nodes", 0.0)
            sam_nodes = bucket.get("sam_nodes", 0.0)
            eagle_matches = bucket.get("eagle_matches", 0.0)
            sam_matches = bucket.get("sam_matches", 0.0)
            strata["depth_{}".format(depth)] = {
                "depth": depth,
                "steps_with_target": int(bucket.get("steps_with_target", 0.0)),
                "eagle_nodes": _safe_div(eagle_nodes, steps),
                "sam_nodes": _safe_div(sam_nodes, steps),
                "eagle_matches": _safe_div(eagle_matches, steps),
                "sam_matches": _safe_div(sam_matches, steps),
                "eagle_match_rate": _safe_div(eagle_matches, eagle_nodes),
                "sam_match_rate": _safe_div(sam_matches, sam_nodes),
                "eagle_avg_score": _safe_div(bucket.get("eagle_score_sum", 0.0), eagle_nodes),
                "sam_avg_score": _safe_div(bucket.get("sam_score_sum", 0.0), sam_nodes),
            }
        return strata

    def depth_strata(self, d_split: int, max_split: int = DEFAULT_MAX_SPLIT) -> List[Dict[str, Any]]:
        d_split = max(int(d_split), 0)
        max_split = max(int(max_split), d_split)
        prefix = {
            "depth": "0-{}".format(d_split),
            "depth_start": 0,
            "depth_end": d_split,
            "owner": "eagle",
            "role": "eagle_prefix",
        }
        tail = {
            "depth": "{}-{}".format(d_split + 1, max_split),
            "depth_start": d_split + 1,
            "depth_end": max_split,
            "owner": "sam",
            "role": "sam_tail",
        }
        if not self.steps:
            return [
                {
                    **prefix,
                    "active_nodes": 0.0,
                    "eagle_nodes": 0.0,
                    "sam_nodes": 0.0,
                    "node_share": 0.0,
                    "matches": 0.0,
                    "match_rate": 0.0,
                    "accepted_nonroot": 0.0,
                    "possible_nonroot": 0.0,
                    "nonroot_acceptance_rate": 0.0,
                    "mat_contribution": 0.0,
                },
                {
                    **tail,
                    "active_nodes": 0.0,
                    "eagle_nodes": 0.0,
                    "sam_nodes": 0.0,
                    "node_share": 0.0,
                    "matches": 0.0,
                    "match_rate": 0.0,
                    "accepted_nonroot": 0.0,
                    "possible_nonroot": 0.0,
                    "nonroot_acceptance_rate": 0.0,
                    "mat_contribution": 0.0,
                },
            ]

        totals = {
            "prefix_eagle_nodes": 0.0,
            "prefix_sam_nodes": 0.0,
            "prefix_matches": 0.0,
            "prefix_accepted": 0.0,
            "prefix_possible": 0.0,
            "tail_eagle_nodes": 0.0,
            "tail_sam_nodes": 0.0,
            "tail_matches": 0.0,
            "tail_accepted": 0.0,
            "tail_possible": 0.0,
        }

        for step in self.steps:
            eagle_depth = self._source_accept_depth(step, "eagle")
            prefix_accepted = min(eagle_depth, d_split)
            sam_extension = self._sam_extension_depth(step, d_split, eagle_depth)
            prefix_possible = min(len(step.acceptance_path), d_split)
            tail_possible = max(min(len(step.acceptance_path), max_split) - d_split, 0)

            totals["prefix_accepted"] += float(prefix_accepted)
            totals["tail_accepted"] += float(min(sam_extension, tail_possible))
            totals["prefix_possible"] += float(prefix_possible)
            totals["tail_possible"] += float(tail_possible)

            for candidate in step.candidates:
                depth = int(candidate.depth)
                if depth <= d_split:
                    key = "prefix_{}_nodes".format(candidate.source)
                    totals[key] = totals.get(key, 0.0) + 1.0
                    if candidate.source == "eagle" and candidate_matches_acceptance(candidate, step):
                        totals["prefix_matches"] += 1.0
                elif depth <= max_split:
                    key = "tail_{}_nodes".format(candidate.source)
                    totals[key] = totals.get(key, 0.0) + 1.0
                    if candidate.source == "sam" and candidate_matches_acceptance(candidate, step):
                        totals["tail_matches"] += 1.0

        steps = len(self.steps)
        prefix_active_nodes = totals["prefix_eagle_nodes"]
        tail_active_nodes = totals["tail_sam_nodes"]
        active_nodes = prefix_active_nodes + tail_active_nodes
        prefix_avg_active = _safe_div(prefix_active_nodes, steps)
        tail_avg_active = _safe_div(tail_active_nodes, steps)
        return [
            {
                **prefix,
                "active_nodes": prefix_avg_active,
                "eagle_nodes": _safe_div(totals["prefix_eagle_nodes"], steps),
                "sam_nodes": _safe_div(totals["prefix_sam_nodes"], steps),
                "node_share": _safe_div(prefix_active_nodes, active_nodes),
                "matches": _safe_div(totals["prefix_matches"], steps),
                "match_rate": _safe_div(totals["prefix_matches"], prefix_active_nodes),
                "accepted_nonroot": _safe_div(totals["prefix_accepted"], steps),
                "possible_nonroot": _safe_div(totals["prefix_possible"], steps),
                "nonroot_acceptance_rate": _safe_div(
                    totals["prefix_accepted"],
                    totals["prefix_possible"],
                ),
                "mat_contribution": 1.0 + _safe_div(totals["prefix_accepted"], steps),
            },
            {
                **tail,
                "active_nodes": tail_avg_active,
                "eagle_nodes": _safe_div(totals["tail_eagle_nodes"], steps),
                "sam_nodes": _safe_div(totals["tail_sam_nodes"], steps),
                "node_share": _safe_div(tail_active_nodes, active_nodes),
                "matches": _safe_div(totals["tail_matches"], steps),
                "match_rate": _safe_div(totals["tail_matches"], tail_active_nodes),
                "accepted_nonroot": _safe_div(totals["tail_accepted"], steps),
                "possible_nonroot": _safe_div(totals["tail_possible"], steps),
                "nonroot_acceptance_rate": _safe_div(
                    totals["tail_accepted"],
                    totals["tail_possible"],
                ),
                "mat_contribution": _safe_div(totals["tail_accepted"], steps),
            },
        ]

    def analyze(self, max_split: int = DEFAULT_MAX_SPLIT) -> Dict[str, Any]:
        sweep_results = self.sweep(max_split)
        optimal_d_split = self.find_optimal_split(max_split)
        oracle_results: Dict[str, Any] = {"eagle_only": self.eagle_only_result()}
        for d_split in sorted(sweep_results):
            oracle_results[_result_key(d_split)] = sweep_results[d_split]
        return {
            "schema_version": 1,
            "steps": len(self.steps),
            "max_split": int(max_split),
            "optimal_d_split": optimal_d_split,
            "oracle_results": oracle_results,
            "depth_strata": self.depth_strata(optimal_d_split, max_split=max_split),
            "depth_details": self.depth_details(),
        }


def _coerce_trace(trace: Any) -> Any:
    if isinstance(trace, (DecodeStep, dict, list)):
        return trace
    return list(trace)


def simulate_leaf_extension(trace: Any, d_split: int) -> Dict[str, Any]:
    """Simulate EAGLE-prefix/SAM-tail oracle MAT for one split."""
    return DepthStratifiedOracle(_coerce_trace(trace)).simulate_split(d_split)


def find_optimal_split(trace: Any) -> int:
    """Return the best ``D_split`` in the default 1..10 sweep."""
    return DepthStratifiedOracle(_coerce_trace(trace)).find_optimal_split(DEFAULT_MAX_SPLIT)


def render_summary(analysis: Dict[str, Any]) -> str:
    oracle_results = analysis["oracle_results"]
    optimal = int(analysis.get("optimal_d_split", 0))
    lines = [
        "Depth-decoupled oracle analysis",
        "steps: {}".format(analysis.get("steps", 0)),
        "optimal_d_split: {}".format(optimal),
        "",
        "{:<22} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
            "method",
            "MAT",
            "gap",
            "eagle_nodes",
            "sam_nodes",
            "prefix_ok",
        ),
    ]
    eagle = oracle_results["eagle_only"]
    lines.append(
        "{:<22} {:>10.4f} {:>10} {:>12.2f} {:>12.2f} {:>12}".format(
            "eagle_only",
            float(eagle.get("mat", 0.0)),
            "+0.00%",
            float(eagle.get("eagle_nodes", 0.0)),
            float(eagle.get("sam_nodes", 0.0)),
            "n/a",
        )
    )
    for key in sorted(
        (key for key in oracle_results if key.startswith("depth_decoupled_d")),
        key=lambda item: int(item.rsplit("d", 1)[1]),
    ):
        row = oracle_results[key]
        gap = row.get("gap")
        lines.append(
            "{:<22} {:>10.4f} {:>10} {:>12.2f} {:>12.2f} {:>12.2%}".format(
                key,
                float(row.get("mat", 0.0)),
                "n/a" if gap is None else "{:+.2%}".format(float(gap)),
                float(row.get("eagle_nodes", 0.0)),
                float(row.get("sam_nodes", 0.0)),
                float(row.get("prefix_success_rate", 0.0)),
            )
        )
    return "\n".join(lines)


def _split_gap_points(analysis: Dict[str, Any]) -> List[Tuple[int, float]]:
    oracle_results = analysis.get("oracle_results", {})
    points: List[Tuple[int, float]] = []
    for key, row in oracle_results.items():
        if not key.startswith("depth_decoupled_d"):
            continue
        gap = row.get("gap")
        if gap is None:
            continue
        d_split = row.get("d_split")
        if d_split is None:
            d_split = key.rsplit("d", 1)[1]
        points.append((int(d_split), float(gap)))
    return sorted(points)


def _format_gap_percent(gap: float) -> str:
    return "{:+.2f}%".format(float(gap) * 100.0)


def write_gap_plot(analysis: Dict[str, Any], output_path: Path) -> Path:
    """Write a dependency-free SVG plot of oracle gap vs D_split."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width = 720
    height = 420
    margin_left = 76
    margin_right = 32
    margin_top = 42
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    points = _split_gap_points(analysis)

    svg: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}" role="img" aria-labelledby="title desc">'.format(
            width,
            height,
        ),
        '<title id="title">Oracle gap vs D_split</title>',
        '<desc id="desc">Depth-decoupled oracle gap for each D_split value.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933} .tick{fill:#52606d;font-size:12px} .label{fill:#323f4b;font-size:13px} .title{font-size:18px;font-weight:700}</style>',
        '<text x="{0}" y="24" text-anchor="middle" class="title">Oracle gap vs D_split</text>'.format(
            width / 2,
        ),
    ]

    if not points:
        svg.extend(
            [
                '<text x="{0}" y="{1}" text-anchor="middle" class="label">No split gap data</text>'.format(
                    width / 2,
                    height / 2,
                ),
                "</svg>",
            ]
        )
        output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
        return output_path

    gaps = [gap for _, gap in points]
    d_values = [d_split for d_split, _ in points]
    min_gap = min([0.0] + gaps)
    max_gap = max([0.0] + gaps)
    span = max_gap - min_gap
    if span == 0.0:
        padding = max(abs(max_gap) * 0.10, 0.01)
    else:
        padding = span * 0.08
    y_min = min_gap - padding
    y_max = max_gap + padding
    y_span = y_max - y_min

    min_d = min(d_values)
    max_d = max(d_values)
    d_span = max(max_d - min_d, 1)

    def x_for(d_split: int) -> float:
        if len(points) == 1:
            return margin_left + plot_width / 2
        return margin_left + ((float(d_split) - min_d) / d_span) * plot_width

    def y_for(gap: float) -> float:
        return margin_top + ((y_max - float(gap)) / y_span) * plot_height

    zero_y = y_for(0.0)
    svg.append(
        '<rect x="{0}" y="{1}" width="{2}" height="{3}" fill="#f8fafc" stroke="#d9e2ec"/>'.format(
            margin_left,
            margin_top,
            plot_width,
            plot_height,
        )
    )

    for index in range(5):
        value = y_min + (y_span * index / 4)
        y_pos = y_for(value)
        svg.append(
            '<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" stroke="#edf2f7"/>'.format(
                margin_left,
                y_pos,
                margin_left + plot_width,
            )
        )
        svg.append(
            '<text x="{0}" y="{1:.2f}" text-anchor="end" dominant-baseline="middle" class="tick">{2}</text>'.format(
                margin_left - 10,
                y_pos,
                escape(_format_gap_percent(value)),
            )
        )

    svg.append(
        '<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" stroke="#9fb3c8" stroke-width="1.5"/>'.format(
            margin_left,
            zero_y,
            margin_left + plot_width,
        )
    )
    svg.append(
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#9fb3c8" stroke-width="1.5"/>'.format(
            margin_left,
            margin_top,
            margin_top + plot_height,
        )
    )
    svg.append(
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#9fb3c8" stroke-width="1.5"/>'.format(
            margin_left,
            margin_top + plot_height,
            margin_left + plot_width,
        )
    )

    bar_width = min(34.0, max(8.0, plot_width / (len(points) * 1.8)))
    line_points: List[str] = []
    for d_split, gap in points:
        x_pos = x_for(d_split)
        y_pos = y_for(gap)
        bar_y = min(y_pos, zero_y)
        bar_height = max(abs(zero_y - y_pos), 1.0)
        fill = "#2f80ed" if gap >= 0 else "#c2410c"
        line_points.append("{:.2f},{:.2f}".format(x_pos, y_pos))
        svg.append(
            '<rect x="{0:.2f}" y="{1:.2f}" width="{2:.2f}" height="{3:.2f}" fill="{4}" opacity="0.32"><title>D_split {5}: {6}</title></rect>'.format(
                x_pos - bar_width / 2,
                bar_y,
                bar_width,
                bar_height,
                fill,
                d_split,
                escape(_format_gap_percent(gap)),
            )
        )

    svg.append(
        '<polyline fill="none" stroke="#1f5f8b" stroke-width="2.5" points="{}"/>'.format(
            " ".join(line_points),
        )
    )

    for d_split, gap in points:
        x_pos = x_for(d_split)
        y_pos = y_for(gap)
        label_y = y_pos - 10 if gap >= 0 else y_pos + 18
        svg.append(
            '<circle cx="{0:.2f}" cy="{1:.2f}" r="4" fill="#1f5f8b"><title>D_split {2}: {3}</title></circle>'.format(
                x_pos,
                y_pos,
                d_split,
                escape(_format_gap_percent(gap)),
            )
        )
        svg.append(
            '<text x="{0:.2f}" y="{1:.2f}" text-anchor="middle" class="tick">{2}</text>'.format(
                x_pos,
                label_y,
                escape(_format_gap_percent(gap)),
            )
        )
        svg.append(
            '<text x="{0:.2f}" y="{1}" text-anchor="middle" class="tick">{2}</text>'.format(
                x_pos,
                margin_top + plot_height + 22,
                d_split,
            )
        )

    svg.append(
        '<text x="{0}" y="{1}" text-anchor="middle" class="label">D_split</text>'.format(
            margin_left + plot_width / 2,
            height - 18,
        )
    )
    svg.append(
        '<text x="18" y="{0}" text-anchor="middle" class="label" transform="rotate(-90 18 {0})">gap vs eagle_only</text>'.format(
            margin_top + plot_height / 2,
        )
    )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output_path


def _default_plot_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name("{}_gap.svg".format(output_path.stem))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-file", action="append", required=True, help="Fusion profile JSON/JSONL trace file.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--plot-output", help="SVG plot output path. Defaults to <output-stem>_gap.svg next to --output.")
    parser.add_argument("--max-split", type=int, default=DEFAULT_MAX_SPLIT, help="Maximum D_split to sweep from 1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.trace_file]
    oracle = DepthStratifiedOracle.from_files(paths)
    if not oracle.steps:
        raise SystemExit("ERROR: no decode steps with candidates found")
    analysis = oracle.analyze(max_split=args.max_split)
    analysis["trace_files"] = [str(path) for path in paths]
    output_path = Path(args.output)
    plot_path = Path(args.plot_output) if args.plot_output else _default_plot_path(output_path)
    analysis["plot_file"] = str(plot_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_gap_plot(analysis, plot_path)
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(render_summary(analysis))
    print("output_json: {}".format(output_path))
    print("plot_svg: {}".format(plot_path))


if __name__ == "__main__":
    main()
