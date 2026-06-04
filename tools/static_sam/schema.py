"""Schema and manifest helpers for Static SAM offline corpora."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"

CANONICAL_FIELDS = [
    "sample_id",
    "domain",
    "source_id",
    "source_name",
    "source_url",
    "license",
    "instruction",
    "input",
    "prompt",
    "source_response",
    "generated_response",
    "response_generator",
    "response_generated_at",
    "is_eval_dataset",
    "eval_exclusion_reason",
    "split",
    "tags",
]

REQUIRED_FIELDS = [
    "sample_id",
    "domain",
    "source_id",
    "prompt",
    "source_response",
    "is_eval_dataset",
]

RESPONSE_FIELDS = {"source_response", "generated_response"}
DEFAULT_DOMAINS = {"medical", "finance"}
DEFAULT_SPLIT = "static_sam_train"


def now_iso() -> str:
    """Return a stable UTC timestamp for manifests."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def normalize_tags(value: Any, domain: str | None = None) -> list[str]:
    tags: list[str] = []
    if isinstance(value, str):
        tags.extend(tag.strip() for tag in value.split(",") if tag.strip())
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, dict)):
        tags.extend(str(tag).strip() for tag in value if str(tag).strip())
    if domain and domain not in tags:
        tags.append(domain)
    return tags


def make_sample(
    *,
    sample_id: str,
    domain: str,
    source_id: str,
    source_name: str,
    source_url: str = "",
    license: str = "unknown",
    instruction: str,
    input_text: str = "",
    prompt: str,
    source_response: str,
    generated_response: str = "",
    response_generator: str = "",
    response_generated_at: str = "",
    is_eval_dataset: bool = False,
    eval_exclusion_reason: str = "",
    split: str = DEFAULT_SPLIT,
    tags: Any = None,
) -> dict[str, Any]:
    return {
        "sample_id": normalize_text(sample_id),
        "domain": normalize_text(domain),
        "source_id": normalize_text(source_id),
        "source_name": normalize_text(source_name),
        "source_url": normalize_text(source_url),
        "license": normalize_text(license),
        "instruction": normalize_text(instruction),
        "input": normalize_text(input_text),
        "prompt": normalize_text(prompt),
        "source_response": normalize_text(source_response),
        "generated_response": normalize_text(generated_response),
        "response_generator": normalize_text(response_generator),
        "response_generated_at": normalize_text(response_generated_at),
        "is_eval_dataset": normalize_bool(is_eval_dataset),
        "eval_exclusion_reason": normalize_text(eval_exclusion_reason),
        "split": normalize_text(split or DEFAULT_SPLIT),
        "tags": normalize_tags(tags, domain),
    }


def has_text(value: Any) -> bool:
    return bool(normalize_text(value).strip())


def count_by(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[normalize_text(row.get(field))] += 1
    counter.pop("", None)
    return dict(sorted(counter.items()))


def build_corpus_manifest(
    corpus_path: str | Path,
    rows: list[dict[str, Any]],
    *,
    validation: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = normalize_text(row.get("source_id"))
        if not source_id:
            continue
        entry = sources.setdefault(
            source_id,
            {
                "domain": normalize_text(row.get("domain")),
                "count": 0,
                "source_name": normalize_text(row.get("source_name")),
                "source_url": normalize_text(row.get("source_url")),
                "license": normalize_text(row.get("license")),
                "is_eval_dataset": normalize_bool(row.get("is_eval_dataset")),
                "eval_exclusion_reason": normalize_text(row.get("eval_exclusion_reason")),
            },
        )
        entry["count"] += 1

    return {
        "corpus_id": Path(corpus_path).name,
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "sample_count": len(rows),
        "domains": count_by(rows, "domain"),
        "sources": dict(sorted(sources.items())),
        "response_fields": {
            "source_response": {
                "available_count": sum(1 for row in rows if has_text(row.get("source_response"))),
            },
            "generated_response": {
                "available_count": sum(1 for row in rows if has_text(row.get("generated_response"))),
                "generator": generation.get("model_name") if generation else None,
            },
        },
        "generation": generation or {},
        "validation": validation or {
            "status": "not_run",
            "strict_no_eval_dataset": False,
            "denied_source_hits": [],
            "errors": [],
            "warnings": [],
        },
    }


def build_artifact_manifest(
    *,
    artifact_id: str,
    corpus_id: str,
    model_name: str,
    tokenizer_name_or_path: str,
    response_field: str,
    cutoff_len: int,
    sample_count: int,
    effective_sample_count: int,
    skipped_count: int,
    truncated_count: int,
    domain_counts: dict[str, int],
    source_counts: dict[str, int],
    strict_no_eval_dataset: bool,
    sam_path: str,
    eos_token_id: int | None,
    tokenizer_vocab_size: int | None,
    smoke_load: bool,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "corpus_id": corpus_id,
        "schema_version": SCHEMA_VERSION,
        "model_name": model_name,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "response_field": response_field,
        "cutoff_len": cutoff_len,
        "sample_count": sample_count,
        "effective_sample_count": effective_sample_count,
        "skipped_count": skipped_count,
        "truncated_count": truncated_count,
        "domain_counts": domain_counts,
        "source_counts": source_counts,
        "strict_no_eval_dataset": strict_no_eval_dataset,
        "sam_path": sam_path,
        "eos_token_id": eos_token_id,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "smoke_load": smoke_load,
        "built_at": now_iso(),
    }
