"""Import local instruction/QA files into a Static SAM canonical corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.prompter import Prompter

from .corpus_io import (
    copy_registry_snapshot,
    dataset_path,
    load_rows,
    save_manifest,
    save_rows,
)
from .registry import (
    DEFAULT_REGISTRY_PATH,
    find_allowed_source,
    load_registry,
)
from .schema import build_corpus_manifest, make_sample, normalize_bool, normalize_text
from .validate_corpus import validate_rows


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"JSON input must be an object or list of objects: {path}")


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_no} must be an object: {path}")
            rows.append(payload)
    return rows


def _load_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Parquet import requires `datasets`: pip install datasets"
        ) from exc
    dataset = load_dataset("parquet", data_files=str(path), split="train")
    return [dict(row) for row in dataset]


def load_local_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_records(input_path)
    if suffix == ".json":
        return _load_json_records(input_path)
    if suffix == ".parquet":
        return _load_parquet_records(input_path)
    raise ValueError(f"Unsupported input extension {suffix!r}; expected .json, .jsonl, or .parquet")


def _field(row: dict[str, Any], field_name: str | None) -> str:
    if not field_name:
        return ""
    return normalize_text(row.get(field_name))


def _build_prompt(
    *,
    instruction: str,
    input_text: str,
    prompt_template_name: str,
    prompter: Prompter | None,
) -> str:
    if prompt_template_name == "none":
        if input_text:
            return f"{instruction}\n\n{input_text}"
        return instruction
    if prompter is None:
        prompter = Prompter(prompt_template_name)
    return prompter.generate_prompt(instruction, input_text)


def import_local(
    *,
    input_path: str | Path,
    output_path: str | Path,
    domain: str,
    source_id: str,
    source_name: str,
    license: str,
    eval_exclusion_reason: str,
    instruction_field: str,
    response_field: str,
    input_field: str | None = None,
    source_url: str = "",
    prompt_field: str | None = None,
    prompt_template_name: str = "vicuna",
    split: str = "static_sam_train",
    tags: str = "",
    is_eval_dataset: bool = False,
    registry_path: str | Path | None = None,
    allow_unregistered_local_source: bool = False,
    replace_source: bool = False,
) -> dict[str, Any]:
    registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    registry = load_registry(registry_path)
    allowed = find_allowed_source(registry, source_id)

    source_name = source_name or normalize_text((allowed or {}).get("source_name")) or source_id
    license = license or normalize_text((allowed or {}).get("license")) or "unknown"
    source_url = source_url or normalize_text((allowed or {}).get("source_url"))
    eval_exclusion_reason = (
        eval_exclusion_reason
        or normalize_text((allowed or {}).get("eval_exclusion_reason"))
    )

    raw_rows = load_local_records(input_path)
    prompter = None if prompt_template_name == "none" else Prompter(prompt_template_name)
    imported_rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_rows):
        instruction = _field(raw, instruction_field)
        input_text = _field(raw, input_field)
        prompt = _field(raw, prompt_field) or _build_prompt(
            instruction=instruction,
            input_text=input_text,
            prompt_template_name=prompt_template_name,
            prompter=prompter,
        )
        imported_rows.append(
            make_sample(
                sample_id=f"{source_id}:{idx:06d}",
                domain=domain,
                source_id=source_id,
                source_name=source_name,
                source_url=source_url,
                license=license,
                instruction=instruction,
                input_text=input_text,
                prompt=prompt,
                source_response=_field(raw, response_field),
                is_eval_dataset=is_eval_dataset,
                eval_exclusion_reason=eval_exclusion_reason,
                split=split,
                tags=tags,
            )
        )

    output_path = Path(output_path)
    existing_rows: list[dict[str, Any]] = []
    if dataset_path(output_path).exists():
        existing_rows = load_rows(output_path)
    if replace_source:
        existing_rows = [row for row in existing_rows if normalize_text(row.get("source_id")) != source_id]
    rows = existing_rows + imported_rows

    report = validate_rows(
        rows,
        registry,
        allow_unregistered_local_source=allow_unregistered_local_source,
    )
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    save_rows(output_path, rows)
    copy_registry_snapshot(registry_path, output_path)
    manifest = build_corpus_manifest(output_path, rows, validation=report)
    save_manifest(output_path, manifest)
    print(f"imported rows: {len(imported_rows)}")
    print(f"total corpus rows: {len(rows)}")
    print(f"domains: {manifest['domains']}")
    print(f"sources: {list(manifest['sources'].keys())}")
    if report["warnings"]:
        print(f"warnings: {len(report['warnings'])}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-name", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license", default="")
    parser.add_argument("--eval-exclusion-reason", default="")
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--response-field", default="answer")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--prompt-template-name", default="vicuna")
    parser.add_argument("--split", default="static_sam_train")
    parser.add_argument("--tags", default="")
    parser.add_argument("--is-eval-dataset", action="store_true")
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--allow-unregistered-local-source", action="store_true")
    parser.add_argument("--replace-source", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_local(
        input_path=args.input,
        output_path=args.output,
        domain=args.domain,
        source_id=args.source_id,
        source_name=args.source_name,
        source_url=args.source_url,
        license=args.license,
        eval_exclusion_reason=args.eval_exclusion_reason,
        instruction_field=args.instruction_field,
        input_field=args.input_field,
        response_field=args.response_field,
        prompt_field=args.prompt_field,
        prompt_template_name=args.prompt_template_name,
        split=args.split,
        tags=args.tags,
        is_eval_dataset=args.is_eval_dataset,
        registry_path=args.registry_path,
        allow_unregistered_local_source=args.allow_unregistered_local_source,
        replace_source=args.replace_source,
    )


if __name__ == "__main__":
    main()
