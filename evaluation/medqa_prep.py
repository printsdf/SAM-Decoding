"""Convert HuggingFace bigbio/med_qa USMLE-style MCQ into the Spec-Bench
``question.jsonl`` format consumed by ``evaluation/eval_llama3.py``.

Output schema per line (per
``.codestable/features/2026-05-24-medqa-vmiss-eval/medqa-vmiss-eval-design.md``
section 1 D6):

    {
        "question_id": int,
        "category": "medqa",
        "turns": [<prompt str ending in "Answer:">]
    }

Prompt template:

    {question}

    A. {options[0]}
    B. {options[1]}
    ...

    Answer:

The script is defensive about the upstream schema because bigbio/med_qa exposes
several configs (``med_qa_en_source``, ``med_qa_en_4options_source``, ...) that
name the options field differently.
"""
import argparse
import json
import os
from typing import Any, Dict, List

_LETTERS = "ABCDEFGHIJ"


def _extract_question(row: Dict[str, Any]) -> str:
    for key in ("question", "stem"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise KeyError("row has no 'question' or 'stem' field: keys={}".format(list(row.keys())))


def _extract_options(row: Dict[str, Any]) -> List[str]:
    for key in ("options", "choices", "option"):
        val = row.get(key)
        if not isinstance(val, list) or not val:
            continue
        if isinstance(val[0], str):
            return [str(opt) for opt in val]
        if isinstance(val[0], dict):
            extracted: List[str] = []
            for d in val:
                text = d.get("value") or d.get("text") or d.get("content")
                if text is None:
                    continue
                extracted.append(str(text))
            if extracted:
                return extracted
    raise KeyError("row has no recognizable options list: keys={}".format(list(row.keys())))


def _format_prompt(question: str, options: List[str]) -> str:
    if len(options) > len(_LETTERS):
        raise ValueError("too many options ({}); only {} letters supported".format(
            len(options), len(_LETTERS)))
    lines = [question, ""]
    for letter, opt in zip(_LETTERS, options):
        lines.append("{}. {}".format(letter, opt))
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_questions", type=int, default=80,
                        help="Number of questions to materialize (default 80).")
    parser.add_argument("--split", type=str, default="test",
                        help="HuggingFace split to load (default 'test').")
    parser.add_argument("--config_name", type=str, default="med_qa_en_source",
                        help="bigbio/med_qa config to load (default 'med_qa_en_source').")
    parser.add_argument("--out_path", type=str,
                        default="evaluation/data/medqa/question.jsonl",
                        help="Destination jsonl path.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "medqa_prep requires `datasets` library: pip install datasets"
        ) from exc

    print("loading bigbio/med_qa config={} split={} ...".format(args.config_name, args.split))
    try:
        ds = load_dataset("bigbio/med_qa", args.config_name, split=args.split, trust_remote_code=True)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" in str(exc):
            raise RuntimeError(
                "bigbio/med_qa is a script-based HuggingFace dataset which the "
                "installed datasets>=3.0 dropped support for. Fix with either:\n"
                "  (a) pip install 'datasets<3.0' (quickest)\n"
                "  (b) switch to a parquet MedQA mirror by editing this script."
            ) from exc
        raise
    total = len(ds)
    take = min(args.num_questions, total)
    print("dataset size {}; writing first {} rows to {}".format(total, take, args.out_path))

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as fout:
        for idx in range(take):
            row = ds[idx]
            question = _extract_question(row)
            options = _extract_options(row)
            prompt = _format_prompt(question, options)
            fout.write(json.dumps({
                "question_id": idx,
                "category": "medqa",
                "turns": [prompt],
            }, ensure_ascii=False) + "\n")

    print("done: {} questions written".format(take))


if __name__ == "__main__":
    main()
