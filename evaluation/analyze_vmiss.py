"""Aggregate per-step diagnosis traces from three MedQA answer files into a
single accept-length / V_miss comparison table.

V_miss definition (matches design section 0 / D3 + ../EAGLE/eagle/model/ea_model.py
:_make_diagnosis_trace_step): on a tree-path decode step where the verifier
rejected at least one draft token, ``verifier_target_reachable == False`` means
the token the verifier wanted is outside the draft model vocabulary (looked up
through Eagle3's t2d buffer). The miss rate is computed over tree-path steps
where ``verifier_target_reachable`` is not None (i.e. excluding accept-to-end
steps that have no rejection point).

The SAM-only group has no draft vocabulary so ``tree_step_vmiss_rate`` is
reported as ``"N/A"``.
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

_REQUIRED_TRACE_KEYS = {
    "step_idx", "path_type", "accept_length",
    "first_rejected_token_id", "first_rejected_reachable",
    "verifier_target_token_id", "verifier_target_reachable",
}


def _iter_records(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError("answer file not found: {}".format(path))
    with open(path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{}:{} not valid JSON: {}".format(path, line_no, exc)
                ) from exc


def _collect_accept_lengths(records: Iterable[Dict[str, Any]]) -> List[int]:
    """Pull accept_lengths from each choice across all questions / choices."""
    acc: List[int] = []
    for rec in records:
        for choice in rec.get("choices", []):
            acc.extend(int(x) for x in choice.get("accept_lengths", []))
    return acc


def _iter_trace_steps(
    records: Iterable[Dict[str, Any]],
    path: str,
    expect_trace: bool,
) -> Iterable[Dict[str, Any]]:
    """Yield each trace step dict; raise if expect_trace=True but field is absent."""
    saw_any_choice = False
    saw_any_trace_field = False
    for rec in records:
        for choice in rec.get("choices", []):
            saw_any_choice = True
            traces = choice.get("diagnosis_traces")
            if traces is None:
                continue
            saw_any_trace_field = True
            for turn_trace in traces:
                for step in turn_trace:
                    if not isinstance(step, dict):
                        print("WARNING analyze_vmiss: non-dict trace step in {}, skipping".format(path),
                              file=sys.stderr)
                        continue
                    missing = _REQUIRED_TRACE_KEYS - set(step.keys())
                    if missing:
                        print("WARNING analyze_vmiss: trace step in {} missing keys {}, skipping".format(
                            path, sorted(missing)), file=sys.stderr)
                        continue
                    yield step
    if expect_trace and saw_any_choice and not saw_any_trace_field:
        raise ValueError(
            "{} has no `diagnosis_traces` field on any choice; was it produced "
            "without --collect_diagnosis_trace?".format(path)
        )


def _analyze_group(
    label: str,
    path: str,
    expect_trace: bool,
) -> Tuple[Dict[str, Any], Tuple[int, int, int]]:
    """Return (table_row, sanity_counts = (total_steps, tree_steps, sequence_steps))."""
    records_for_accept = list(_iter_records(path))
    accept_lengths = _collect_accept_lengths(records_for_accept)
    mean_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0

    total_steps = 0
    tree_steps = 0
    sequence_steps = 0
    miss_numer = 0
    miss_denom = 0
    if expect_trace:
        for step in _iter_trace_steps(records_for_accept, path, expect_trace=True):
            total_steps += 1
            if step["path_type"] == "tree":
                tree_steps += 1
                vt_reachable = step["verifier_target_reachable"]
                if vt_reachable is not None:
                    miss_denom += 1
                    if vt_reachable is False:
                        miss_numer += 1
            elif step["path_type"] == "sequence":
                sequence_steps += 1
            # other path_type values are ignored from the breakdown

    if not expect_trace:
        return (
            {
                "group": label,
                "mean_accept_length": "{:.3f}".format(mean_accept),
                "tree_steps": "N/A",
                "tree_step_vmiss_rate": "N/A",
            },
            (0, 0, 0),
        )

    if miss_denom == 0:
        vmiss_str = "N/A (no rejected tree step)"
    else:
        vmiss_str = "{:.1%}".format(miss_numer / miss_denom)

    return (
        {
            "group": label,
            "mean_accept_length": "{:.3f}".format(mean_accept),
            "tree_steps": str(tree_steps),
            "tree_step_vmiss_rate": vmiss_str,
        },
        (total_steps, tree_steps, sequence_steps),
    )


def _format_table(rows: List[Dict[str, str]]) -> str:
    header = "| group | mean_accept_length | tree_steps | tree_step_vmiss_rate |"
    sep = "|---|---|---|---|"
    body = "\n".join(
        "| {} | {} | {} | {} |".format(
            r["group"], r["mean_accept_length"], r["tree_steps"], r["tree_step_vmiss_rate"],
        )
        for r in rows
    )
    return "\n".join([header, sep, body])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure_eagle3", required=True,
                        help="Path to pure_eagle3.jsonl (with diagnosis_traces).")
    parser.add_argument("--samd_eagle3", required=True,
                        help="Path to samd_eagle3.jsonl (with diagnosis_traces).")
    parser.add_argument("--sam_only", required=True,
                        help="Path to sam_only.jsonl (no diagnosis_traces).")
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    sanity: List[Tuple[str, Tuple[int, int, int]]] = []
    for label, path, expect in (
        ("pure_eagle3", args.pure_eagle3, True),
        ("samd_eagle3", args.samd_eagle3, True),
        ("sam_only", args.sam_only, False),
    ):
        row, counts = _analyze_group(label, path, expect)
        rows.append(row)
        sanity.append((label, counts))

    print(_format_table(rows))
    print()
    for label, (total, tree, seq) in sanity:
        print("sanity[{}]: total_steps={}, tree_steps={}, sequence_steps={}".format(
            label, total, tree, seq))


if __name__ == "__main__":
    main()
