#!/usr/bin/env python3
"""Profile the naive EAGLE3 + SAM fusion path.

Runs ``evaluation.inference_samd`` under cProfile with
``fusion_mode=naive``, ``tree_fusion=none``, and threshold 5 by default, then
prints an actionable breakdown of the generated profile stats.
"""

from __future__ import annotations

import argparse
import os
import pstats
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


FuncKey = Tuple[str, int, str]


@dataclass(frozen=True)
class FunctionStat:
    func: FuncKey
    primitive_calls: int
    total_calls: int
    self_time: float
    cumulative_time: float


@dataclass(frozen=True)
class FunctionGroup:
    label: str
    predicates: Tuple[Callable[[FuncKey], bool], ...]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_filename(filename: str) -> str:
    return filename.replace("\\", "/")


def func_name(func: FuncKey) -> str:
    return func[2]


def file_contains(*fragments: str) -> Callable[[FuncKey], bool]:
    return lambda func: any(fragment in normalize_filename(func[0]) for fragment in fragments)


def name_is(*names: str) -> Callable[[FuncKey], bool]:
    return lambda func: func_name(func) in names


def name_contains(*fragments: str) -> Callable[[FuncKey], bool]:
    return lambda func: any(fragment in func_name(func) for fragment in fragments)


FUSION_GROUPS: Tuple[FunctionGroup, ...] = (
    FunctionGroup("fuse_eagle_sam_naive", (name_is("fuse_eagle_sam_naive"),)),
    FunctionGroup("parse_eagle_tree", (name_is("parse_eagle_tree"),)),
    FunctionGroup("parse_sam_sequence", (name_is("parse_sam_sequence"),)),
    FunctionGroup("normalize_scores", (name_is("_normalization_groups", "_normalize_score", "_normalize_by_source"),)),
    FunctionGroup("merge_and_dedup", (name_is("merge_and_dedup"),)),
    FunctionGroup("sort_by_score", (name_is("sort_by_score"),)),
    FunctionGroup("sort_depth_first", (name_is("sort_depth_first"),)),
    FunctionGroup("truncate_with_ancestors", (name_is("truncate_with_ancestors", "_ancestor_keys"),)),
    FunctionGroup("build_tree_buffers", (name_is("build_tree_buffers", "_order_prefix_closed_nodes"),)),
    FunctionGroup("TreeSpec buffer conversion", (file_contains("samd/tree_model/fusion.py"),)),
    FunctionGroup("source_counts", (name_is("source_counts"),)),
    FunctionGroup("fusion metadata helpers", (name_is("_safe_mean", "_count_duplicates", "_node_sources_by_tree_index"),)),
    FunctionGroup("fusion tensor/list conversion", (name_is("_squeeze_tree_tokens", "_squeeze_logprobs"),)),
)


def is_fusion_function(func: FuncKey) -> bool:
    filename = normalize_filename(func[0])
    if "samd/fusion/" in filename or "samd/tree_model/fusion.py" in filename:
        return True
    return any(any(predicate(func) for predicate in group.predicates) for group in FUSION_GROUPS)


def component_for(func: FuncKey) -> str:
    filename = normalize_filename(func[0])
    name = func_name(func)

    if name in ("from_pretrained", "_load_pretrained_model") or "modeling_utils.py" in filename:
        return "model_loading"
    if is_fusion_function(func):
        return "fusion"
    if "samd/tree_model/eagle3/" in filename or name in ("topK_genrate",):
        return "eagle3_draft"
    if "samd/sam/" in filename or ("samd/" in filename and name in ("lookup", "gen_draft_raw")):
        return "sam_lookup"
    if "samd/utils.py" in filename or "samd/draft.py" in filename:
        return "candidate_generation"
    if "samd/samd_model.py" in filename or "samd/model_patch/" in filename:
        return "verifier_generation"
    if "evaluation/" in filename or "fastchat/" in filename:
        return "evaluation_io"
    if "torch/" in filename or "transformers/" in filename or "site-packages/torch" in filename:
        return "torch_transformers_runtime"
    return "other"


def is_profile_wrapper(func: FuncKey, root: Path) -> bool:
    filename = normalize_filename(func[0])
    name = func_name(func)
    script_path = normalize_filename(str(root / "scripts/profile_naive_fusion.py"))
    if filename == "~" and name in (
        "<built-in method builtins.exec>",
        "<method 'disable' of '_lsprof.Profiler' objects>",
    ):
        return True
    if filename == "<string>":
        return True
    if filename == script_path:
        return True
    if "runpy" in filename or "cProfile.py" in filename or "profile.py" in filename:
        return True
    return name in ("<module>", "_run_code", "_run_module_code", "run_module")


def pct(seconds: float, total_runtime: float) -> float:
    if total_runtime <= 0:
        return 0.0
    return seconds / total_runtime * 100.0


def format_func(func: FuncKey, root: Path) -> str:
    filename, line_no, name = func
    try:
        display = str(Path(filename).resolve().relative_to(root))
    except (OSError, ValueError):
        display = filename
    return "{} ({}:{})".format(name, display, line_no)


def iter_function_stats(stats: pstats.Stats) -> List[FunctionStat]:
    rows = []
    for func, values in stats.stats.items():
        primitive_calls, total_calls, self_time, cumulative_time, _callers = values
        rows.append(
            FunctionStat(
                func=func,
                primitive_calls=primitive_calls,
                total_calls=total_calls,
                self_time=self_time,
                cumulative_time=cumulative_time,
            )
        )
    return rows


def top_functions(
    rows: Iterable[FunctionStat],
    key: str,
    limit: int,
    root: Path,
    hide_profile_wrappers: bool,
) -> List[FunctionStat]:
    if key == "cumulative":
        sort_key = lambda row: row.cumulative_time
    elif key == "self":
        sort_key = lambda row: row.self_time
    else:
        raise ValueError("unsupported sort key: {}".format(key))

    filtered = [
        row
        for row in rows
        if not hide_profile_wrappers or not is_profile_wrapper(row.func, root)
    ]
    return sorted(filtered, key=sort_key, reverse=True)[:limit]


def group_matches(group: FunctionGroup, func: FuncKey) -> bool:
    return any(predicate(func) for predicate in group.predicates)


def aggregate_fusion_groups(rows: Iterable[FunctionStat]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for group in FUSION_GROUPS:
        matched = [row for row in rows if group_matches(group, row.func)]
        result[group.label] = {
            "self_time": sum(row.self_time for row in matched),
            "cumulative_time": sum(row.cumulative_time for row in matched),
            "calls": sum(row.total_calls for row in matched),
        }
    return result


def aggregate_components(rows: Iterable[FunctionStat]) -> Dict[str, float]:
    components: Dict[str, float] = {}
    for row in rows:
        component = component_for(row.func)
        components[component] = components.get(component, 0.0) + row.self_time
    return components


def build_command(args: argparse.Namespace, stats_file: Path, answer_file: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        str(stats_file),
        "-m",
        "evaluation.inference_samd",
        "--template",
        args.template,
        "--model-type",
        args.model_type,
        "--model-path",
        args.model_path,
        "--model-id",
        args.model_id,
        "--bench-name",
        args.bench_name,
        "--answer-file",
        str(answer_file),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--num-choices",
        str(args.num_choices),
        "--num-gpus-per-model",
        str(args.num_gpus_per_model),
        "--num-gpus-total",
        str(args.num_gpus_total),
        "--temperature",
        str(args.temperature),
        "--dtype",
        args.dtype,
        "--samd_n_predicts",
        str(args.samd_n_predicts),
        "--samd_len_threshold",
        str(args.samd_len_threshold),
        "--samd_len_bias",
        str(args.samd_len_bias),
        "--tree_method",
        "eagle3",
        "--tree_fusion",
        "none",
        "--fusion_mode",
        "naive",
        "--fusion_max_draft_tokens",
        str(args.fusion_max_draft_tokens),
        "--fusion_dedup_strategy",
        args.fusion_dedup_strategy,
        "--fusion_truncate_strategy",
        args.fusion_truncate_strategy,
        "--tree_model_path",
        args.tree_model_path,
        "--eagle3_total_token",
        str(args.eagle3_total_token),
        "--eagle3_depth",
        str(args.eagle3_depth),
        "--eagle3_top_k",
        str(args.eagle3_top_k),
        "--attn_implementation",
        args.attn_implementation,
        "--max_cache_len",
        str(args.max_cache_len),
    ]
    if args.question_begin is not None:
        command.extend(["--question-begin", str(args.question_begin)])
    if args.question_end is not None:
        command.extend(["--question-end", str(args.question_end)])
    if args.sam_path:
        command.extend(["--sam_path", args.sam_path])
    if args.samd_tree_path:
        command.extend(["--samd_tree_path", args.samd_tree_path])
    return command


def build_env(root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    pythonpath_parts = [str(root)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def run_profile_command(command: Sequence[str], env: Dict[str, str]) -> float:
    print("START cProfile run")
    print("Command: {}".format(shlex.join(command)))
    start = time.perf_counter()
    subprocess.run(command, env=env, check=True)
    elapsed = time.perf_counter() - start
    print("DONE cProfile run in {:.2f} seconds".format(elapsed))
    return elapsed


def render_top_section(
    title: str,
    rows: Sequence[FunctionStat],
    total_runtime: float,
    root: Path,
    time_attr: str,
) -> List[str]:
    lines = [title]
    if not rows:
        lines.append("  (no functions recorded)")
        return lines
    for index, row in enumerate(rows, start=1):
        seconds = row.cumulative_time if time_attr == "cumulative_time" else row.self_time
        lines.append(
            "  {}. {}: {:.1f}% ({:.3f}s, {} calls)".format(
                index,
                format_func(row.func, root),
                pct(seconds, total_runtime),
                seconds,
                row.total_calls,
            )
        )
    return lines


def render_fusion_breakdown(
    groups: Dict[str, Dict[str, float]],
    total_runtime: float,
) -> List[str]:
    lines = ["Fusion-specific breakdown:"]
    for label, values in groups.items():
        cumulative = values["cumulative_time"]
        self_time = values["self_time"]
        calls = int(values["calls"])
        lines.append(
            "  - {}: {:.1f}% cumulative ({:.3f}s), {:.1f}% self ({:.3f}s), {} calls".format(
                label,
                pct(cumulative, total_runtime),
                cumulative,
                pct(self_time, total_runtime),
                self_time,
                calls,
            )
        )
    return lines


def render_component_breakdown(components: Dict[str, float], total_runtime: float) -> List[str]:
    lines = ["Major component breakdown (self time, non-overlapping):"]
    for label, seconds in sorted(components.items(), key=lambda item: item[1], reverse=True):
        lines.append("  - {}: {:.1f}% ({:.3f}s)".format(label, pct(seconds, total_runtime), seconds))
    return lines


def render_recommendation(
    fusion_groups: Dict[str, Dict[str, float]],
    components: Dict[str, float],
    rows: Sequence[FunctionStat],
    total_runtime: float,
    threshold_pct: float,
    root: Path,
) -> List[str]:
    high_fusion = [
        label
        for label, values in fusion_groups.items()
        if pct(values["cumulative_time"], total_runtime) > threshold_pct
    ]
    fusion_self_pct = pct(components.get("fusion", 0.0), total_runtime)
    non_wrapper_rows = [row for row in rows if not is_profile_wrapper(row.func, root)]
    high_cumulative = [
        row
        for row in non_wrapper_rows
        if pct(row.cumulative_time, total_runtime) > threshold_pct
    ]
    high_self = [
        row
        for row in non_wrapper_rows
        if pct(row.self_time, total_runtime) > threshold_pct
    ]
    high_cumulative = sorted(high_cumulative, key=lambda row: row.cumulative_time, reverse=True)
    high_self = sorted(high_self, key=lambda row: row.self_time, reverse=True)
    fusion_category_hot = fusion_self_pct > threshold_pct
    optimization_needed = bool(high_cumulative) or bool(high_self) or bool(high_fusion) or fusion_category_hot
    fusion_optimization_needed = bool(high_fusion) or fusion_category_hot

    lines = ["Recommendation:"]
    lines.append(
        "  [{}] Code optimization needed (any function > {:.1f}%)".format(
            "x" if optimization_needed else " ",
            threshold_pct,
        )
    )
    lines.append(
        "  [{}] No obvious bottleneck, proceed to Phase 3.2".format(
            " " if optimization_needed else "x"
        )
    )
    if high_fusion:
        lines.append("  Fusion hotspots above threshold: {}".format(", ".join(high_fusion)))
    elif fusion_self_pct > threshold_pct:
        lines.append("  Fusion self-time category is {:.1f}% of runtime.".format(fusion_self_pct))
    elif high_self:
        top_names = ", ".join(format_func(row.func, root) for row in high_self[:3])
        lines.append(
            "  Self-time hotspots above threshold are not fusion-specific: {}.".format(top_names)
        )
    elif high_cumulative:
        top_names = ", ".join(format_func(row.func, root) for row in high_cumulative[:3])
        lines.append(
            "  Cumulative hotspots above threshold are not fusion-specific: {}.".format(top_names)
        )
    else:
        lines.append("  No single cumulative function exceeded the threshold.")
    if fusion_optimization_needed:
        lines.append("  Fusion-path decision: optimize naive fusion before Phase 3.2.")
    else:
        lines.append("  Fusion-path decision: no fusion-specific hotspot above threshold.")
    return lines


def render_report(
    stats: pstats.Stats,
    measured_wall_time: Optional[float],
    stats_file: Path,
    answer_file: Path,
    report_file: Path,
    args: argparse.Namespace,
    root: Path,
) -> str:
    rows = iter_function_stats(stats)
    total_runtime = float(stats.total_tt)
    top_cumulative = top_functions(
        rows,
        key="cumulative",
        limit=args.top_limit,
        root=root,
        hide_profile_wrappers=not args.include_profile_wrappers,
    )
    top_self = top_functions(
        rows,
        key="self",
        limit=args.top_limit,
        root=root,
        hide_profile_wrappers=not args.include_profile_wrappers,
    )
    fusion_groups = aggregate_fusion_groups(rows)
    components = aggregate_components(rows)

    question_range = "[{}, {})".format(args.question_begin, args.question_end)
    lines = [
        "=== Profiling Results ===",
        "Total runtime: {:.3f} seconds".format(total_runtime),
    ]
    if measured_wall_time is not None:
        lines.append("Measured wall time: {:.3f} seconds".format(measured_wall_time))
    lines.extend(
        [
            "Question range: {}".format(question_range),
            "Profile stats: {}".format(stats_file),
            "Answer file: {}".format(answer_file),
            "Report file: {}".format(report_file),
            "Config: fusion_mode=naive, tree_fusion=none, threshold={}, max_cache_len={}".format(
                args.samd_len_threshold,
                args.max_cache_len,
            ),
            "",
        ]
    )
    lines.extend(
        render_top_section(
            "Top bottlenecks (cumulative time):",
            top_cumulative,
            total_runtime,
            root,
            "cumulative_time",
        )
    )
    lines.append("")
    lines.extend(
        render_top_section(
            "Top functions by self time:",
            top_self,
            total_runtime,
            root,
            "self_time",
        )
    )
    lines.append("")
    lines.extend(render_fusion_breakdown(fusion_groups, total_runtime))
    lines.append("")
    lines.extend(render_component_breakdown(components, total_runtime))
    lines.append("")
    lines.extend(
        render_recommendation(
            fusion_groups,
            components,
            rows,
            total_runtime,
            args.bottleneck_threshold_pct,
            root,
        )
    )
    return "\n".join(lines) + "\n"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", "--model_path", dest="model_path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument(
        "--tree-model-path",
        "--tree_model_path",
        dest="tree_model_path",
        default=os.environ.get("TREE_MODEL_PATH"),
    )
    parser.add_argument("--sam-path", "--sam_path", dest="sam_path", default=os.environ.get("SAM_PATH") or None)
    parser.add_argument(
        "--samd-tree-path",
        "--samd_tree_path",
        dest="samd_tree_path",
        default=os.environ.get("SAMD_TREE_PATH") or None,
    )
    parser.add_argument("--template", default=os.environ.get("TEMPLATE", "llama3"), choices=["vicuna", "llama3"])
    parser.add_argument(
        "--model-type",
        "--model_type",
        dest="model_type",
        default=os.environ.get("MODEL_TYPE", "llama3"),
        choices=["vicuna", "llama3"],
    )
    parser.add_argument("--bench-name", "--bench_name", dest="bench_name", default=os.environ.get("BENCH_NAME", "mt_bench"))
    parser.add_argument("--question-begin", "--question_begin", dest="question_begin", type=int, default=0)
    parser.add_argument("--question-end", "--question_end", dest="question_end", type=int, default=None)
    parser.add_argument("--num-questions", "--num_questions", dest="num_questions", type=positive_int, default=10)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)
    parser.add_argument("--answer-file", "--answer_file", dest="answer_file", default=None)
    parser.add_argument("--stats-file", "--stats_file", dest="stats_file", default=None)
    parser.add_argument("--report-file", "--report_file", dest="report_file", default=None)
    parser.add_argument("--model-id", "--model_id", dest="model_id", default=None)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", "512")))
    parser.add_argument("--max-cache-len", "--max_cache_len", dest="max_cache_len", type=int, default=int(os.environ.get("MAX_CACHE_LEN", "4096")))
    parser.add_argument("--allow-large-cache", action="store_true")
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "float16"))
    parser.add_argument("--num-choices", "--num_choices", dest="num_choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", "--num_gpus_per_model", dest="num_gpus_per_model", type=int, default=1)
    parser.add_argument("--num-gpus-total", "--num_gpus_total", dest="num_gpus_total", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samd-n-predicts", "--samd_n_predicts", dest="samd_n_predicts", type=int, default=40)
    parser.add_argument("--samd-len-threshold", "--samd_len_threshold", dest="samd_len_threshold", type=int, default=5)
    parser.add_argument("--samd-len-bias", "--samd_len_bias", dest="samd_len_bias", type=int, default=5)
    parser.add_argument("--eagle3-total-token", "--eagle3_total_token", dest="eagle3_total_token", type=int, default=int(os.environ.get("EAGLE3_TOTAL_TOKEN", "60")))
    parser.add_argument("--eagle3-depth", "--eagle3_depth", dest="eagle3_depth", type=int, default=int(os.environ.get("EAGLE3_DEPTH", "7")))
    parser.add_argument("--eagle3-top-k", "--eagle3_top_k", dest="eagle3_top_k", type=int, default=int(os.environ.get("EAGLE3_TOP_K", "10")))
    parser.add_argument("--fusion-max-draft-tokens", "--fusion_max_draft_tokens", dest="fusion_max_draft_tokens", type=int, default=60)
    parser.add_argument(
        "--fusion-dedup-strategy",
        "--fusion_dedup_strategy",
        dest="fusion_dedup_strategy",
        default="max_score",
        choices=["max_score", "sum_score", "keep_both"],
    )
    parser.add_argument(
        "--fusion-truncate-strategy",
        "--fusion_truncate_strategy",
        dest="fusion_truncate_strategy",
        default="score",
        choices=["score", "depth_first"],
    )
    parser.add_argument("--attn-implementation", "--attn_implementation", dest="attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--top-limit", "--top_limit", dest="top_limit", type=int, default=20)
    parser.add_argument("--bottleneck-threshold-pct", "--bottleneck_threshold_pct", dest="bottleneck_threshold_pct", type=float, default=5.0)
    parser.add_argument("--skip-run", "--skip_run", dest="skip_run", action="store_true", help="Analyze an existing --stats-file without running evaluation.")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print the cProfile command without running it.")
    parser.add_argument("--include-profile-wrappers", action="store_true", help="Show runpy/cProfile wrapper frames in top-function tables.")
    args = parser.parse_args()

    if args.question_end is None:
        args.question_end = args.question_begin + args.num_questions
    if args.question_end <= args.question_begin:
        parser.error("--question-end must be greater than --question-begin")
    if args.max_cache_len > 4096 and not args.allow_large_cache:
        parser.error("--max-cache-len must be <= 4096 unless --allow-large-cache is set")
    if not args.skip_run:
        missing = []
        if not args.model_path:
            missing.append("--model-path or MODEL_PATH")
        if not args.tree_model_path:
            missing.append("--tree-model-path or TREE_MODEL_PATH")
        if missing:
            parser.error("missing required runtime path(s): {}".format(", ".join(missing)))

    output_dir = Path(args.output_dir) if args.output_dir else Path("evaluation/data") / args.bench_name / "profile_naive_fusion"
    range_tag = "q{}_{}".format(args.question_begin, args.question_end)
    if args.model_id is None:
        args.model_id = "naive_logprob_profile_{}".format(range_tag)
    if args.answer_file is None:
        args.answer_file = str(output_dir / "{}.jsonl".format(args.model_id))
    if args.stats_file is None:
        args.stats_file = str(output_dir / "{}.stats".format(args.model_id))
    if args.report_file is None:
        args.report_file = str(output_dir / "{}_report.txt".format(args.model_id))
    return args


def main() -> None:
    args = parse_args()
    root = repo_root()
    os.chdir(root)

    answer_file = Path(args.answer_file)
    stats_file = Path(args.stats_file)
    report_file = Path(args.report_file)
    for path in (answer_file, stats_file, report_file):
        path.parent.mkdir(parents=True, exist_ok=True)

    measured_wall_time: Optional[float] = None
    command = build_command(args, stats_file, answer_file)
    if args.dry_run:
        print("Command: {}".format(shlex.join(command)))
        return
    if not args.skip_run:
        measured_wall_time = run_profile_command(command, build_env(root))
    elif not stats_file.exists():
        raise SystemExit("ERROR: --skip-run requested but stats file does not exist: {}".format(stats_file))

    stats = pstats.Stats(str(stats_file))
    report = render_report(stats, measured_wall_time, stats_file, answer_file, report_file, args, root)
    report_file.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
