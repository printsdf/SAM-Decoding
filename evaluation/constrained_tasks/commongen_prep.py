"""Prepare CommonGen as ``question.jsonl`` for the Llama-3 eval loop.

Each output line keeps the existing benchmark schema consumed by
``evaluation/eval_llama3.py`` and adds ``concepts`` metadata used later by the
quality analyzer.
"""
import argparse
import json
import os
from typing import Any, Dict, Iterable, List


PROMPT_TEMPLATE = """Write a sentence using all the given set of concepts.
Example 1:
Concepts: ['rest', 'family', 'grass']
Sentence: family with baby girl resting on a grass in a park.

Example 2:
Concepts: ['boil', 'kettle', 'water']
Sentence: electric kettle in which the water is boiled.

Example 3:
Concepts: {concepts}
Sentence:"""


def extract_concepts(row: Dict[str, Any]) -> List[str]:
    """Return the concept list from common CommonGen dataset schemas."""
    for key in ("concepts", "concept_set", "concept_set_idx"):
        value = row.get(key)
        if isinstance(value, list):
            concepts = [str(item).strip() for item in value if str(item).strip()]
            if concepts:
                return concepts
    raise KeyError("row has no non-empty concept list: keys={}".format(list(row.keys())))


def build_prompt(concepts: Iterable[str]) -> str:
    concept_list = [str(item) for item in concepts]
    return PROMPT_TEMPLATE.format(concepts=repr(concept_list))


def write_questions(rows: Iterable[Dict[str, Any]], out_path: str, limit: int) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows:
            if written >= limit:
                break
            concepts = extract_concepts(row)
            fout.write(json.dumps({
                "question_id": written,
                "category": "commongen",
                "turns": [build_prompt(concepts)],
                "concepts": concepts,
            }, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="allenai/common_gen",
                        help="HuggingFace dataset id (default: allenai/common_gen).")
    parser.add_argument("--split", default="validation",
                        help="HuggingFace split to load (default: validation).")
    parser.add_argument("--num_questions", type=int, default=200,
                        help="Number of examples to materialize (default: 200).")
    parser.add_argument("--out_path", default="evaluation/data/commongen/question.jsonl",
                        help="Destination question.jsonl path.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "commongen_prep requires `datasets` library: pip install 'datasets<3.0'"
        ) from exc

    print("loading {} split={} ...".format(args.dataset_name, args.split))
    ds = load_dataset(args.dataset_name, split=args.split)
    take = min(args.num_questions, len(ds))
    print("dataset size {}; writing first {} rows to {}".format(len(ds), take, args.out_path))
    written = write_questions((ds[idx] for idx in range(len(ds))), args.out_path, take)
    print("done: {} questions written".format(written))


if __name__ == "__main__":
    main()
