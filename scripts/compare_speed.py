#!/usr/bin/env python3
"""Compare MAT / tokens-per-second across answer jsonl files.

Usage:
    python3 scripts/compare_speed.py BASELINE.jsonl METHOD.jsonl [MORE.jsonl ...]

The first file is the reference; every file reports absolute stats and
MAT gain / speedup relative to it. Records with missing turns (ERROR
outputs keep their wall_time) are included as-is.
"""

import argparse
import json
import os
import sys


def stats(path):
    tokens = steps = 0
    wall = 0.0
    accept = []
    questions = 0
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            choice = json.loads(line)["choices"][0]
            tokens += sum(choice["new_tokens"])
            steps += sum(choice["decoding_steps"])
            wall += sum(choice["wall_time"])
            accept.extend(choice.get("accept_lengths", []))
            questions += 1
    if questions == 0 or steps == 0 or wall == 0.0:
        raise SystemExit(f"{path}: empty or malformed answer file")
    return {
        "questions": questions,
        "tokens": tokens,
        "steps": steps,
        "wall_s": wall,
        "mat": tokens / steps,
        "mean_accept": sum(accept) / len(accept) if accept else float("nan"),
        "tok_per_s": tokens / wall,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help="answer jsonl files; first = baseline")
    args = parser.parse_args()

    rows = []
    for path in args.files:
        label = os.path.splitext(os.path.basename(path))[0]
        rows.append((label, stats(path)))

    base = rows[0][1]
    header = (
        f"{'model':40s} {'q':>4s} {'tokens':>8s} {'steps':>7s} {'MAT':>7s} "
        f"{'accept':>7s} {'tok/s':>8s} {'time_s':>8s} {'MAT+%':>7s} {'speedup':>8s}"
    )
    print(header)
    print("-" * len(header))
    for label, s in rows:
        mat_gain = (s["mat"] / base["mat"] - 1) * 100
        speedup = s["tok_per_s"] / base["tok_per_s"]
        print(
            f"{label:40s} {s['questions']:>4d} {s['tokens']:>8d} {s['steps']:>7d} "
            f"{s['mat']:>7.4f} {s['mean_accept']:>7.4f} {s['tok_per_s']:>8.2f} "
            f"{s['wall_s']:>8.1f} {mat_gain:>+7.2f} {speedup:>8.4f}"
        )

    if len(rows) > 1:
        counts = {s["questions"] for _, s in rows}
        if len(counts) > 1:
            print(f"WARNING: question counts differ across files: {sorted(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
