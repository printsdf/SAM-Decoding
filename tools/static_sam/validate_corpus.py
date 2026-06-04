"""Validate a Static SAM canonical corpus before artifact construction."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .corpus_io import load_json, load_rows, manifest_path, save_manifest
from .registry import (
    DEFAULT_REGISTRY_PATH,
    allowed_domains,
    find_allowed_source,
    find_denied_source,
    load_registry,
)
from .schema import (
    CANONICAL_FIELDS,
    DEFAULT_DOMAINS,
    REQUIRED_FIELDS,
    RESPONSE_FIELDS,
    build_corpus_manifest,
    has_text,
    normalize_bool,
    normalize_text,
)


def _entry_bool(entry: dict[str, Any], key: str, default: bool) -> bool:
    if key not in entry:
        return default
    return normalize_bool(entry.get(key))


def validate_rows(
    rows: list[dict[str, Any]],
    registry: dict[str, Any],
    *,
    allow_unregistered_local_source: bool = False,
    require_response_field: str | None = None,
) -> dict[str, Any]:
    if require_response_field and require_response_field not in RESPONSE_FIELDS:
        raise ValueError(
            f"Unknown response field {require_response_field!r}; expected one of {sorted(RESPONSE_FIELDS)}"
        )

    errors: list[str] = []
    warnings: list[str] = []
    denied_source_hits: list[dict[str, str]] = []
    unregistered_sources: set[str] = set()
    domains = allowed_domains(registry, DEFAULT_DOMAINS)
    sample_ids: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        prefix = f"row {idx}"
        missing_fields = [field for field in REQUIRED_FIELDS if field not in row]
        for field in missing_fields:
            errors.append(f"{prefix}: missing required field {field}")

        for field in CANONICAL_FIELDS:
            if field not in row:
                warnings.append(f"{prefix}: missing canonical field {field}")

        sample_id = normalize_text(row.get("sample_id"))
        if sample_id:
            sample_ids[sample_id] += 1
        else:
            errors.append(f"{prefix}: sample_id is empty")

        domain = normalize_text(row.get("domain"))
        if not domain:
            errors.append(f"{prefix}: domain is empty")
        elif domain not in domains:
            errors.append(f"{prefix}: unsupported domain {domain!r}; allowed domains are {sorted(domains)}")

        source_id = normalize_text(row.get("source_id"))
        if not source_id:
            errors.append(f"{prefix}: source_id is empty")
            continue

        denied = find_denied_source(registry, source_id)
        if denied is not None:
            reason = normalize_text(denied.get("reason")) or "denied source"
            errors.append(f"{prefix}: source_id {source_id!r} is denied: {reason}")
            denied_source_hits.append({"source_id": source_id, "reason": reason})

        allowed = find_allowed_source(registry, source_id)
        if allowed is None:
            if allow_unregistered_local_source:
                unregistered_sources.add(source_id)
            else:
                errors.append(f"{prefix}: source_id {source_id!r} is not in allowed_sources")
        else:
            if _entry_bool(allowed, "is_eval_dataset", False):
                errors.append(f"{prefix}: allowed source {source_id!r} is marked is_eval_dataset=True")
            if not _entry_bool(allowed, "allow_static_sam_build", True):
                errors.append(f"{prefix}: source_id {source_id!r} has allow_static_sam_build=false")

        if normalize_bool(row.get("is_eval_dataset")):
            errors.append(f"{prefix}: is_eval_dataset=True cannot enter Static SAM corpus")

        if not has_text(row.get("prompt")):
            errors.append(f"{prefix}: prompt is empty")
        if not has_text(row.get("source_response")):
            errors.append(f"{prefix}: source_response is empty")
        if require_response_field and not has_text(row.get(require_response_field)):
            errors.append(f"{prefix}: selected response field {require_response_field} is empty")

        if not has_text(row.get("eval_exclusion_reason")):
            warnings.append(f"{prefix}: eval_exclusion_reason is empty")
        if not has_text(row.get("license")) or normalize_text(row.get("license")).lower() == "unknown":
            warnings.append(f"{prefix}: license is empty or unknown")
        if not has_text(row.get("source_url")):
            warnings.append(f"{prefix}: source_url is empty")
        if not has_text(row.get("generated_response")):
            warnings.append(f"{prefix}: generated_response is empty")

    for sample_id, count in sample_ids.items():
        if count > 1:
            errors.append(f"duplicate sample_id {sample_id!r} appears {count} times")

    return {
        "status": "passed" if not errors else "failed",
        "strict_no_eval_dataset": not errors,
        "denied_source_hits": denied_source_hits,
        "unregistered_sources": sorted(unregistered_sources),
        "errors": errors,
        "warnings": warnings,
    }


def validate_corpus(
    corpus_path: str | Path,
    *,
    registry_path: str | Path | None = None,
    allow_unregistered_local_source: bool = False,
    require_response_field: str | None = None,
    persist_manifest: bool = True,
) -> dict[str, Any]:
    rows = load_rows(corpus_path)
    registry = load_registry(registry_path)
    report = validate_rows(
        rows,
        registry,
        allow_unregistered_local_source=allow_unregistered_local_source,
        require_response_field=require_response_field,
    )
    if persist_manifest:
        existing_manifest = load_json(manifest_path(corpus_path))
        save_manifest(
            corpus_path,
            build_corpus_manifest(
                corpus_path,
                rows,
                validation=report,
                generation=existing_manifest.get("generation") or None,
            ),
        )
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"validation status: {report['status']}")
    print(f"errors: {len(report['errors'])}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    print(f"warnings: {len(report['warnings'])}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--allow-unregistered-local-source", action="store_true")
    parser.add_argument("--require-response-field", choices=sorted(RESPONSE_FIELDS), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_corpus(
        args.corpus_path,
        registry_path=args.registry_path,
        allow_unregistered_local_source=args.allow_unregistered_local_source,
        require_response_field=args.require_response_field,
    )
    _print_report(report)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
