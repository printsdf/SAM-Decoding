"""Trace I/O for same-trace oracle analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from evaluation.oracle.types import DecodeStep


def warn(message: str) -> None:
    print("WARNING: {}".format(message), file=sys.stderr)


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
