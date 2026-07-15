"""Multi-method same-trace oracle analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from evaluation.oracle.load import warn
from evaluation.oracle.select import (
    default_budget,
    select_budgeted,
    select_eagle3_only,
    select_perfect,
    select_sam_sequence_graft,
    select_source_balanced,
    selected_mat,
)
from evaluation.oracle.types import DecodeStep, MethodResult


def analyze_steps(
    steps: Sequence[DecodeStep],
    top_k: Optional[int],
    node_budget: Optional[int],
    gap_baseline: str,
    depth_max_split: int = 10,
) -> List[MethodResult]:
    if not steps:
        return []

    selectors = {
        "eagle3_only": lambda step: select_eagle3_only(step, top_k),
        "sam_sequence_graft": lambda step: select_sam_sequence_graft(step, top_k),
        "perfect": lambda step: select_perfect(step, top_k),
        "budgeted": lambda step: select_budgeted(step, default_budget(step, node_budget or top_k)),
        "source_balanced": lambda step: select_source_balanced(step, default_budget(step, top_k)),
    }
    aggregates: Dict[str, Dict[str, float]] = {
        method: {"mat": 0.0, "nodes": 0.0} for method in selectors
    }
    for step in steps:
        for method, selector in selectors.items():
            selected = selector(step)
            aggregates[method]["mat"] += selected_mat(step, selected)
            aggregates[method]["nodes"] += float(len(selected))

    baseline_total = aggregates.get(gap_baseline, {}).get("mat", 0.0)
    baseline_mat = baseline_total / len(steps) if baseline_total > 0 else 0.0
    results: List[MethodResult] = []
    for method in ("eagle3_only", "sam_sequence_graft", "perfect", "budgeted", "source_balanced"):
        mat = aggregates[method]["mat"] / len(steps)
        nodes = aggregates[method]["nodes"] / len(steps)
        gap = (mat - baseline_mat) / baseline_mat if baseline_mat > 0 else None
        results.append(
            MethodResult(
                method=method,
                mat=mat,
                nodes=nodes,
                mat_per_node=(mat / nodes if nodes > 0 else 0.0),
                steps=len(steps),
                oracle_gap=gap,
            )
        )
    try:
        from evaluation.oracle.depth_decoupled import DepthStratifiedOracle

        depth_oracle = DepthStratifiedOracle(list(steps))
        depth_analysis = depth_oracle.analyze(max_split=depth_max_split)
        optimal_d_split = depth_analysis.get("optimal_d_split")
        depth_result = depth_analysis.get("oracle_results", {}).get(
            "depth_decoupled_d{}".format(optimal_d_split)
        )
        if depth_result is not None:
            mat = float(depth_result.get("mat", 0.0))
            nodes = float(depth_result.get("nodes", 0.0))
            gap = (mat - baseline_mat) / baseline_mat if baseline_mat > 0 else None
            results.append(
                MethodResult(
                    method="depth_decoupled",
                    mat=mat,
                    nodes=nodes,
                    mat_per_node=(mat / nodes if nodes > 0 else 0.0),
                    steps=len(steps),
                    oracle_gap=gap,
                )
            )
    except Exception as exc:
        warn("depth_decoupled oracle skipped: {}".format(exc))
    return results


def render_table(results: Sequence[MethodResult], gap_baseline: str) -> str:
    lines = [
        "Oracle fusion analysis",
        "Gap baseline: {}".format(gap_baseline),
        "",
        "{:<20} {:>10} {:>10} {:>12} {:>12}".format(
            "method",
            "MAT",
            "nodes",
            "MAT/node",
            "oracle_gap",
        ),
    ]
    for row in results:
        gap = "n/a" if row.oracle_gap is None else "{:+.2%}".format(row.oracle_gap)
        lines.append(
            "{:<20} {:>10.4f} {:>10.2f} {:>12.5f} {:>12}".format(
                row.method,
                row.mat,
                row.nodes,
                row.mat_per_node,
                gap,
            )
        )
    return "\n".join(lines)
