"""Convert HuggingFace ``lavita/MedQuAD`` (NIH National Library of Medicine
patient question + clinician answer corpus) into the Spec-Bench
``question.jsonl`` format consumed by ``evaluation/eval_llama3.py``.

Output schema per line:

    {
        "question_id": int,
        "category": "medquad",
        "turns": [<patient question str, free-form, no MCQ structure>]
    }

The prompt template is intentionally free-form: the patient question is fed
verbatim as ``turns[0]`` so the chat template wraps it as a user turn and the
model answers naturally. No multiple-choice scaffolding; see
``.trellis/spec/backend/evaluation-protocols.md`` for the free-form V_miss
benchmark rationale.
"""
import argparse
import json
import os
from typing import Any, Dict


def _extract_question(row: Dict[str, Any]) -> str:
    for key in ("question", "Question", "stem"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise KeyError("row has no 'question' field: keys={}".format(list(row.keys())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_questions", type=int, default=200,
                        help="Number of questions to materialize (default 200).")
    parser.add_argument("--split", type=str, default="train",
                        help="HuggingFace split to load (default 'train').")
    parser.add_argument("--dataset_name", type=str, default="lavita/MedQuAD",
                        help="HuggingFace dataset id (default 'lavita/MedQuAD').")
    parser.add_argument("--out_path", type=str,
                        default="evaluation/data/medquad/question.jsonl",
                        help="Destination jsonl path.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "medquad_prep requires `datasets` library: pip install 'datasets<3.0'"
        ) from exc

    print("loading {} split={} ...".format(args.dataset_name, args.split))
    ds = load_dataset(args.dataset_name, split=args.split)
    total = len(ds)
    take = min(args.num_questions, total)
    print("dataset size {}; writing first {} rows to {}".format(total, take, args.out_path))

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    written = 0
    with open(args.out_path, "w", encoding="utf-8") as fout:
        for idx in range(total):
            if written >= take:
                break
            row = ds[idx]
            try:
                question = _extract_question(row)
            except KeyError:
                continue
            fout.write(json.dumps({
                "question_id": written,
                "category": "medquad",
                "turns": [question],
            }, ensure_ascii=False) + "\n")
            written += 1

    print("done: {} questions written".format(written))


if __name__ == "__main__":
    main()
