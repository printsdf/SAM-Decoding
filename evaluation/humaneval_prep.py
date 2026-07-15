"""Convert standard HumanEval benchmark (164 problems) into the Spec-Bench
``question.jsonl`` format consumed by ``evaluation/eval_llama3.py``.

Output schema per line (see ``.trellis/spec/backend/evaluation-protocols.md``):

    {
        "question_id": int,
        "category": "code",
        "turns": ["Complete the code I provided.\n\n<prompt>"],
        "reference": ["<canonical_solution>"]
    }

The script tries importing from the ``human_eval`` package first, then falls
back to downloading from the official OpenAI HumanEval GitHub repository.
"""
import argparse
import hashlib
import json
import os
from typing import Any, Dict, List


def _load_humaneval_from_package() -> List[Dict[str, Any]]:
    """Load HumanEval from the human_eval package if installed."""
    try:
        from human_eval.data import read_problems
    except ImportError as exc:
        raise ImportError(
            "human_eval package not found. Install with: pip install human-eval"
        ) from exc

    problems = read_problems()
    return [{"task_id": k, **v} for k, v in problems.items()]


def _load_humaneval_from_github() -> List[Dict[str, Any]]:
    """Download HumanEval from GitHub as fallback."""
    import urllib.request

    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    print("downloading HumanEval from {} ...".format(url))

    try:
        import gzip
        with urllib.request.urlopen(url) as response:
            data = gzip.decompress(response.read())
    except Exception as exc:
        raise RuntimeError(
            "Failed to download HumanEval from GitHub. "
            "Try installing the package instead: pip install human-eval"
        ) from exc

    problems = []
    for line in data.decode("utf-8").strip().split("\n"):
        if line.strip():
            problems.append(json.loads(line))
    return problems


def _extract_prompt(row: Dict[str, Any]) -> str:
    """Extract prompt from HumanEval row."""
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise KeyError("row has no 'prompt' field: keys={}".format(list(row.keys())))
    return prompt.strip()


def _extract_solution(row: Dict[str, Any]) -> str:
    """Extract canonical solution from HumanEval row."""
    solution = row.get("canonical_solution")
    if not isinstance(solution, str):
        raise KeyError("row has no 'canonical_solution' field: keys={}".format(list(row.keys())))
    return solution


def _format_prompt(prompt: str) -> str:
    """Format prompt with instruction prefix."""
    return "Complete the code I provided.\n\n{}".format(prompt)


def _compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str,
                        default="evaluation/data/humaneval/question.jsonl",
                        help="Destination jsonl path (default: evaluation/data/humaneval/question.jsonl).")
    parser.add_argument("--use-github", action="store_true",
                        help="Force download from GitHub instead of using installed package.")
    args = parser.parse_args()

    # Try loading from package first, then GitHub
    try:
        if args.use_github:
            raise ImportError("--use-github flag set")
        print("loading HumanEval from human_eval package ...")
        problems = _load_humaneval_from_package()
    except ImportError:
        problems = _load_humaneval_from_github()

    if len(problems) != 164:
        print("WARNING: expected 164 problems, got {}".format(len(problems)))

    print("loaded {} problems; writing to {} ...".format(len(problems), args.output))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for idx, row in enumerate(problems):
            try:
                prompt = _extract_prompt(row)
                solution = _extract_solution(row)
                formatted_prompt = _format_prompt(prompt)

                fout.write(json.dumps({
                    "question_id": idx,
                    "category": "code",
                    "turns": [formatted_prompt],
                    "reference": [solution],
                }, ensure_ascii=False) + "\n")
            except KeyError as exc:
                print("WARNING: skipping problem {} due to missing field: {}".format(idx, exc))
                continue

    print("done: {} questions written to {}".format(len(problems), args.output))

    # Compute and print SHA256 hash
    sha256 = _compute_sha256(args.output)
    print("SHA256: {}".format(sha256))


if __name__ == "__main__":
    main()
