"""Run a small HumanEval comparison for naive dual-draft fusion.

This script intentionally keeps model execution in the existing benchmark
runner. It builds the three commands needed for a first-pass comparison:

1. EAGLE3-only SAMD
2. SAM-only routing baseline
3. Naive EAGLE3 + SAM fusion

Actual model evaluation should be run on the remote model environment described
in AGENTS.md. Use --dry_run locally to inspect commands without loading models.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List


def _base_samd_command(
    args: argparse.Namespace,
    model_id: str,
    answer_file: Path,
    len_threshold: int | None = None,
) -> List[str]:
    command = [
        "python",
        "-m",
        "evaluation.inference_samd",
        "--template",
        args.template,
        "--model-type",
        args.model_type,
        "--model-path",
        args.model_path,
        "--model-id",
        model_id,
        "--bench-name",
        args.bench_name,
        "--answer-file",
        str(answer_file),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--samd_n_predicts",
        str(args.samd_n_predicts),
        "--samd_len_threshold",
        str(args.samd_len_threshold if len_threshold is None else len_threshold),
        "--samd_len_bias",
        str(args.samd_len_bias),
        "--tree_method",
        "eagle3",
        "--tree_model_path",
        args.tree_model_path,
        "--eagle3_total_token",
        str(args.eagle3_total_token),
        "--eagle3_depth",
        str(args.eagle3_depth),
        "--eagle3_top_k",
        str(args.eagle3_top_k),
        "--max_cache_len",
        str(args.max_cache_len),
    ]
    if args.sam_path is not None:
        command.extend(["--sam_path", args.sam_path])
    if args.question_begin is not None:
        command.extend(["--question-begin", str(args.question_begin)])
    if args.question_end is not None:
        command.extend(["--question-end", str(args.question_end)])
    return command


def _base_sam_only_command(args: argparse.Namespace, model_id: str, answer_file: Path) -> List[str]:
    command = [
        "python",
        "-m",
        "evaluation.inference_sam_only",
        "--template",
        args.template,
        "--model-type",
        args.model_type,
        "--model-path",
        args.model_path,
        "--model-id",
        model_id,
        "--bench-name",
        args.bench_name,
        "--answer-file",
        str(answer_file),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--samd_max_predicts",
        str(args.samd_n_predicts),
        "--samd_len_bias",
        str(args.samd_len_bias),
        "--max_cache_len",
        str(args.max_cache_len),
    ]
    if args.sam_path is not None:
        command.extend(["--sam_path", args.sam_path])
    if args.question_begin is not None:
        command.extend(["--question-begin", str(args.question_begin)])
    if args.question_end is not None:
        command.extend(["--question-end", str(args.question_end)])
    return command


def build_commands(args: argparse.Namespace) -> Dict[str, List[str]]:
    output_dir = Path(args.output_dir)
    return {
        "eagle3_only": _base_samd_command(
            args,
            "eagle3-only",
            output_dir / "eagle3_only.jsonl",
            len_threshold=args.eagle_only_len_threshold,
        )
        + ["--tree_fusion", "none"],
        "sam_only": _base_sam_only_command(
            args,
            "sam-only",
            output_dir / "sam_only.jsonl",
        ),
        "naive_fusion": _base_samd_command(
            args,
            "naive-fusion",
            output_dir / "naive_fusion.jsonl",
        )
        + [
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
        ],
    }


def run_command(name: str, command: List[str], dry_run: bool) -> None:
    print("=== {} ===".format(name))
    print(shlex.join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def parse_fusion_stats(answer_file: Path) -> Dict[str, float]:
    if not answer_file.exists():
        return {}
    stats = {
        "records": 0,
        "tokens_per_second": 0.0,
        "mat": 0.0,
    }
    for line in answer_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        stats["records"] += 1
        choice = record.get("choices", [{}])[0]
        new_tokens = sum(choice.get("new_tokens") or [])
        wall_time = sum(choice.get("wall_time") or [])
        if wall_time > 0:
            stats["tokens_per_second"] += new_tokens / wall_time
        accept_lengths = choice.get("accept_lengths") or []
        if accept_lengths:
            stats["mat"] += sum(accept_lengths) / len(accept_lengths)
    if stats["records"]:
        stats["tokens_per_second"] /= stats["records"]
        stats["mat"] /= stats["records"]
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tree_model_path", required=True)
    parser.add_argument("--sam_path", default=None)
    parser.add_argument("--template", default="llama3", choices=["vicuna", "llama3"])
    parser.add_argument("--model_type", default="llama3", choices=["vicuna", "llama3"])
    parser.add_argument("--bench_name", default="humaneval")
    parser.add_argument("--output_dir", default="evaluation/data/humaneval/model_answer/naive_fusion")
    parser.add_argument("--question_begin", type=int, default=None)
    parser.add_argument("--question_end", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_cache_len", type=int, default=2048)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--samd_n_predicts", type=int, default=40)
    parser.add_argument("--samd_len_threshold", type=int, default=5)
    parser.add_argument("--eagle_only_len_threshold", type=int, default=1000000)
    parser.add_argument("--samd_len_bias", type=int, default=5)
    parser.add_argument("--eagle3_total_token", type=int, default=60)
    parser.add_argument("--eagle3_depth", type=int, default=7)
    parser.add_argument("--eagle3_top_k", type=int, default=10)
    parser.add_argument("--fusion_max_draft_tokens", type=int, default=60)
    parser.add_argument(
        "--fusion_dedup_strategy",
        default="max_score",
        choices=["max_score", "sum_score", "keep_both"],
    )
    parser.add_argument(
        "--fusion_truncate_strategy",
        default="score",
        choices=["score", "depth_first"],
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    for name, command in commands.items():
        run_command(name, command, args.dry_run)

    if not args.dry_run:
        print("=== summary ===")
        all_stats = {}
        for name in commands:
            answer_file = output_dir / "{}.jsonl".format(name)
            all_stats[name] = parse_fusion_stats(answer_file)
        baseline_tps = all_stats.get("eagle3_only", {}).get("tokens_per_second", 0.0)
        for name, stats in all_stats.items():
            if baseline_tps > 0 and stats:
                stats["speedup_vs_eagle3"] = stats["tokens_per_second"] / baseline_tps
            print("{}: {}".format(name, json.dumps(stats, sort_keys=True)))


if __name__ == "__main__":
    main()
