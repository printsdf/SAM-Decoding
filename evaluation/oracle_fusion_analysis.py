#!/usr/bin/env python3
"""Offline oracle analysis for EAGLE3 + SAM fusion traces.

Expected per-step schema:

{
  "step": 0,
  "candidates": [
    {"source": "eagle", "token": 42, "depth": 1, "score": -0.2, "accepted": true}
  ],
  "acceptance_path": [42],
  "stats": {"mat": 1.0, "eagle_nodes": 40, "sam_nodes": 20}
}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class OracleCandidate:
    source: str
    token: int
    depth: int
    score: float
    accepted: bool
    path: Tuple[int, ...]
    token_path: Tuple[int, ...]
    tree_index: Optional[int]
    parent_index: Optional[int] = None
    local_logprob: Optional[float] = None
    rank_among_siblings: Optional[int] = None
    sibling_margin: Optional[float] = None
    cumulative_path_logprob: Optional[float] = None
    sam_match_length: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OracleCandidate":
        source = str(data.get("source", ""))
        if source not in ("eagle", "sam"):
            raise ValueError("candidate source must be 'eagle' or 'sam': {}".format(source))
        path = _int_tuple(data.get("path", []))
        token_path = _int_tuple(data.get("token_path", []))
        if "token" in data:
            token = int(data["token"])
        elif token_path:
            token = int(token_path[-1])
        else:
            raise KeyError("token")
        if not path and len(token_path) > 1:
            path = token_path[:-1]
        depth = data.get("depth")
        if depth is None:
            if path:
                depth = len(path)
            elif token_path:
                depth = max(len(token_path) - 1, 1)
            else:
                depth = 1
        tree_index = data.get("tree_index", data.get("index", data.get("tree_idx")))
        return cls(
            source=source,
            token=token,
            depth=int(depth),
            score=float(data.get("score", 0.0)),
            accepted=_bool_value(data.get("accepted", False)),
            path=path,
            token_path=token_path,
            tree_index=(int(tree_index) if tree_index is not None else None),
            parent_index=_optional_int(data.get("parent_index")),
            local_logprob=_optional_float(data.get("local_logprob")),
            rank_among_siblings=_optional_int(data.get("rank_among_siblings")),
            sibling_margin=_optional_float(data.get("sibling_margin")),
            cumulative_path_logprob=_optional_float(data.get("cumulative_path_logprob")),
            sam_match_length=_optional_int(data.get("sam_match_length")),
        )


@dataclass(frozen=True)
class DecodeStep:
    step: int
    candidates: List[OracleCandidate]
    acceptance_path: List[int]
    stats: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    first_rejected_depth: Optional[int] = None
    first_rejected_tree_index: Optional[int] = None
    first_rejected_parent_index: Optional[int] = None
    first_rejected_parent_path: Optional[Tuple[int, ...]] = None
    has_rejection_boundary_label: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any], fallback_step: int) -> "DecodeStep":
        candidates = [
            OracleCandidate.from_dict(item)
            for item in data.get("candidates", [])
        ]
        acceptance_path = [int(token) for token in data.get("acceptance_path", [])]
        stats = dict(data.get("stats", {}))
        label_fields_present = (
            "first_rejected_depth" in data
            or "first_rejected_depth" in stats
        )
        rejected_parent_path = data.get(
            "first_rejected_parent_path",
            stats.get("first_rejected_parent_path"),
        )
        metadata = dict(data.get("metadata", {}))
        for key in ("question_id", "question_index", "question_idx", "question_number"):
            if key in data and key not in metadata:
                metadata[key] = data[key]
        return cls(
            step=int(data.get("step", fallback_step)),
            candidates=candidates,
            acceptance_path=acceptance_path,
            stats=stats,
            metadata=metadata,
            first_rejected_depth=_optional_int(
                data.get("first_rejected_depth", stats.get("first_rejected_depth"))
            ),
            first_rejected_tree_index=_optional_int(
                data.get("first_rejected_tree_index", stats.get("first_rejected_tree_index"))
            ),
            first_rejected_parent_index=_optional_int(
                data.get("first_rejected_parent_index", stats.get("first_rejected_parent_index"))
            ),
            first_rejected_parent_path=(
                _int_tuple(rejected_parent_path)
                if rejected_parent_path is not None
                else None
            ),
            has_rejection_boundary_label=label_fields_present,
        )


@dataclass(frozen=True)
class MethodResult:
    method: str
    mat: float
    nodes: float
    mat_per_node: float
    steps: int
    oracle_gap: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "mat": self.mat,
            "nodes": self.nodes,
            "mat_per_node": self.mat_per_node,
            "steps": self.steps,
            "oracle_gap": self.oracle_gap,
        }


def warn(message: str) -> None:
    print("WARNING: {}".format(message), file=sys.stderr)


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _int_tuple(value: Any) -> Tuple[int, ...]:
    if value is None:
        return ()
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    flattened: List[int] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            flattened.extend(_int_tuple(item))
        else:
            flattened.append(int(item))
    return tuple(flattened)


def _load_json_objects(path: Path) -> List[Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        objects = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError as exc:
                warn("{}:{} malformed JSON skipped: {}".format(path, line_no, exc))
        return objects


def _extract_step_dicts(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        steps: List[Dict[str, Any]] = []
        for item in obj:
            steps.extend(_extract_step_dicts(item))
        return steps
    if not isinstance(obj, dict):
        return []
    if "candidates" in obj:
        return [obj]
    for key in ("steps", "trace", "traces", "decode_traces", "oracle_traces"):
        value = obj.get(key)
        if isinstance(value, list):
            return _extract_step_dicts(value)
    if "choices" in obj:
        steps = []
        for choice in obj.get("choices", []):
            steps.extend(_extract_step_dicts(choice))
        return steps
    return []


def coerce_decode_steps(trace: Any) -> List[DecodeStep]:
    if isinstance(trace, DecodeStep):
        return [trace]
    if isinstance(trace, (list, tuple)):
        steps: List[DecodeStep] = []
        for item in trace:
            if isinstance(item, DecodeStep):
                steps.append(item)
            else:
                for step_dict in _extract_step_dicts(item):
                    steps.append(DecodeStep.from_dict(step_dict, fallback_step=len(steps)))
        return steps
    if isinstance(trace, dict) and "candidates" in trace:
        return [DecodeStep.from_dict(trace, fallback_step=0)]
    steps: List[DecodeStep] = []
    for step_dict in _extract_step_dicts(trace):
        steps.append(DecodeStep.from_dict(step_dict, fallback_step=len(steps)))
    return steps


def load_decode_traces(paths: Sequence[Path]) -> List[DecodeStep]:
    steps: List[DecodeStep] = []
    for path in paths:
        before = len(steps)
        for obj in _load_json_objects(path):
            for step_dict in _extract_step_dicts(obj):
                try:
                    step = DecodeStep.from_dict(step_dict, fallback_step=len(steps))
                except (KeyError, TypeError, ValueError) as exc:
                    warn("{} malformed step skipped: {}".format(path, exc))
                    continue
                if not step.candidates:
                    warn("{} step {} has no candidates; skipped".format(path, step.step))
                    continue
                steps.append(step)
        if len(steps) == before:
            warn("{} produced no oracle decode steps".format(path))
    return steps


def candidate_nonroot_path(candidate: OracleCandidate) -> Tuple[int, ...]:
    if candidate.depth <= 0:
        return ()
    if candidate.token_path:
        if len(candidate.token_path) >= candidate.depth + 1:
            return tuple(candidate.token_path[-candidate.depth :])
        return tuple(candidate.token_path)
    if candidate.path:
        parent_nonroot = tuple(candidate.path[1:])
        parent_depth = max(candidate.depth - 1, 0)
        if parent_depth == 0:
            parent_nonroot = ()
        elif len(parent_nonroot) > parent_depth:
            parent_nonroot = parent_nonroot[-parent_depth:]
        return parent_nonroot + (int(candidate.token),)
    return ()


def candidate_matches_acceptance(candidate: OracleCandidate, step: DecodeStep) -> bool:
    if not candidate.accepted:
        return False
    if not step.acceptance_path:
        return True
    nonroot_path = candidate_nonroot_path(candidate)
    if nonroot_path:
        if len(nonroot_path) > len(step.acceptance_path):
            return False
        return list(nonroot_path) == step.acceptance_path[: len(nonroot_path)]
    index = candidate.depth - 1
    return 0 <= index < len(step.acceptance_path) and step.acceptance_path[index] == candidate.token


def _candidate_truth(candidate: OracleCandidate, step: DecodeStep) -> bool:
    return candidate_matches_acceptance(candidate, step)


def _rank_key(step: DecodeStep, candidate: OracleCandidate) -> Tuple[int, int, float, int]:
    truth = 1 if _candidate_truth(candidate, step) else 0
    return (-truth, int(candidate.depth), -float(candidate.score), int(candidate.token))


def _apply_limit(candidates: List[OracleCandidate], limit: Optional[int]) -> List[OracleCandidate]:
    if limit is None:
        return candidates
    return candidates[: max(int(limit), 0)]


def select_perfect(step: DecodeStep, limit: Optional[int]) -> List[OracleCandidate]:
    """Upper-bound selector: true-accepted path candidates first."""
    return _apply_limit(sorted(step.candidates, key=lambda item: _rank_key(step, item)), limit)


def select_budgeted(step: DecodeStep, node_budget: int) -> List[OracleCandidate]:
    """Oracle selector that maximizes accepted-token payoff per selected node."""
    budget = max(int(node_budget), 0)
    ranked = sorted(step.candidates, key=lambda item: _rank_key(step, item))
    return ranked[:budget]


def select_source_balanced(
    step: DecodeStep,
    limit: int,
    min_sam_ratio: float = 0.30,
    max_sam_ratio: float = 0.70,
) -> List[OracleCandidate]:
    """Oracle selector with an approximate 30-70% SAM source-ratio constraint."""
    limit = max(int(limit), 0)
    if limit <= 0:
        return []

    ranked = sorted(step.candidates, key=lambda item: _rank_key(step, item))
    min_sam = int(math.ceil(limit * min_sam_ratio))
    max_sam = int(math.floor(limit * max_sam_ratio))
    selected: List[OracleCandidate] = []
    used = set()

    def choose(source: Optional[str]) -> bool:
        sam_count = sum(1 for item in selected if item.source == "sam")
        for index, candidate in enumerate(ranked):
            if index in used:
                continue
            if source is not None and candidate.source != source:
                continue
            if candidate.source == "sam" and sam_count >= max_sam:
                continue
            used.add(index)
            selected.append(candidate)
            return True
        return False

    while len(selected) < limit:
        sam_count = sum(1 for item in selected if item.source == "sam")
        if sam_count < min_sam and choose("sam"):
            continue
        if choose(None):
            continue
        break
    return selected


def select_eagle3_only(step: DecodeStep, limit: Optional[int]) -> List[OracleCandidate]:
    return _apply_limit([item for item in step.candidates if item.source == "eagle"], limit)


def select_sam_sequence_graft(step: DecodeStep, limit: Optional[int]) -> List[OracleCandidate]:
    eagle = [item for item in step.candidates if item.source == "eagle"]
    sam = [item for item in step.candidates if item.source == "sam"]
    return _apply_limit(eagle + sam, limit)


def selected_mat(step: DecodeStep, selected: Iterable[OracleCandidate]) -> float:
    selected = list(selected)
    # SAMD accept-length metrics count the verifier's root/start token for each
    # decode step. Candidate records are non-root nodes, so add that token here
    # to keep oracle MAT comparable with evaluation accept_lengths.
    root_token = 1.0
    if step.acceptance_path:
        accepted_by_depth: Dict[int, set] = {}
        for candidate in selected:
            if _candidate_truth(candidate, step):
                accepted_by_depth.setdefault(candidate.depth, set()).add(candidate.token)
        accepted = 0
        for depth, token in enumerate(step.acceptance_path, start=1):
            if token not in accepted_by_depth.get(depth, set()):
                break
            accepted += 1
        return root_token + float(accepted)
    return root_token + float(sum(1 for candidate in selected if candidate.accepted))


def _default_budget(step: DecodeStep, fallback: Optional[int]) -> int:
    if fallback is not None:
        return int(fallback)
    stats_budget = int(step.stats.get("eagle_nodes", 0)) + int(step.stats.get("sam_nodes", 0))
    return stats_budget if stats_budget > 0 else len(step.candidates)


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
        "budgeted": lambda step: select_budgeted(step, _default_budget(step, node_budget or top_k)),
        "source_balanced": lambda step: select_source_balanced(step, _default_budget(step, top_k)),
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
        from evaluation.oracle_depth_decoupled import DepthStratifiedOracle

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
            from evaluation.oracle_depth_decoupled import DepthStratifiedOracle

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
