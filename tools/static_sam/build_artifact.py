"""Build tokenizer-specific Static SAM artifacts from a canonical corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .corpus_io import load_rows, save_json
from .registry import DEFAULT_REGISTRY_PATH
from .schema import RESPONSE_FIELDS, build_artifact_manifest, count_by, normalize_text
from .validate_corpus import validate_corpus


BuildSamFn = Callable[[list[list[int]], int], Any]
DumpSamFn = Callable[[str, Any], None]
LoadSamFn = Callable[[str], Any]


def artifact_manifest_path(sam_path: str | Path) -> Path:
    return Path(sam_path).with_suffix(".manifest.json")


def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "build_artifact requires `transformers`: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(model_name)


def _load_sam_runtime() -> tuple[BuildSamFn, DumpSamFn, LoadSamFn]:
    try:
        from samd import build_sam, dump_sam, load_sam
    except ImportError as exc:
        raise ImportError(
            "build_artifact requires samd runtime dependencies such as torch/transformers"
        ) from exc
    return build_sam, dump_sam, load_sam


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, padding=False, return_tensors=None)
    if isinstance(encoded, dict):
        input_ids = encoded.get("input_ids", [])
    else:
        input_ids = encoded
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _tokenizer_len(tokenizer: Any) -> int | None:
    try:
        return len(tokenizer)
    except TypeError:
        return None


def build_artifact(
    *,
    corpus_path: str | Path,
    model_name: str,
    text_response_field: str,
    cutoff_len: int,
    sam_path: str | Path,
    registry_path: str | Path | None = None,
    on_too_long: str = "skip",
    allow_unregistered_local_source: bool = False,
    tokenizer: Any | None = None,
    build_sam_fn: BuildSamFn | None = None,
    dump_sam_fn: DumpSamFn | None = None,
    load_sam_fn: LoadSamFn | None = None,
) -> dict[str, Any]:
    if text_response_field not in RESPONSE_FIELDS:
        raise ValueError(
            f"Unknown response field {text_response_field!r}; expected one of {sorted(RESPONSE_FIELDS)}"
        )
    if cutoff_len <= 0:
        raise ValueError("cutoff_len must be positive")
    if on_too_long not in {"skip", "truncate"}:
        raise ValueError("on_too_long must be 'skip' or 'truncate'")

    validation = validate_corpus(
        corpus_path,
        registry_path=registry_path or DEFAULT_REGISTRY_PATH,
        allow_unregistered_local_source=allow_unregistered_local_source,
        require_response_field=text_response_field,
        persist_manifest=True,
    )
    if validation["errors"]:
        for error in validation["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    if tokenizer is None:
        tokenizer = _load_tokenizer(model_name)
    if build_sam_fn is None or dump_sam_fn is None or load_sam_fn is None:
        default_build_sam, default_dump_sam, default_load_sam = _load_sam_runtime()
        build_sam_fn = build_sam_fn or default_build_sam
        dump_sam_fn = dump_sam_fn or default_dump_sam
        load_sam_fn = load_sam_fn or default_load_sam

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is required to build Static SAM")

    rows = load_rows(corpus_path)
    batch_tokens: list[list[int]] = []
    effective_rows: list[dict[str, Any]] = []
    skipped_count = 0
    truncated_count = 0

    for row in rows:
        text = normalize_text(row.get("prompt")) + normalize_text(row.get(text_response_field))
        input_ids = _tokenize(tokenizer, text)
        if len(input_ids) > cutoff_len:
            if on_too_long == "skip":
                skipped_count += 1
                continue
            input_ids = input_ids[:cutoff_len]
            truncated_count += 1
        if not input_ids:
            skipped_count += 1
            continue
        batch_tokens.append(input_ids)
        effective_rows.append(row)

    if not batch_tokens:
        raise ValueError("No effective samples remained after tokenization")

    vocab_size = _tokenizer_len(tokenizer)
    if vocab_size is not None:
        for token_id in range(vocab_size):
            batch_tokens.append([token_id])

    sam_path = Path(sam_path)
    sam_path.parent.mkdir(parents=True, exist_ok=True)
    sam = build_sam_fn(batch_tokens, int(eos_token_id))
    dump_sam_fn(str(sam_path), sam)

    smoke_load = False
    manifest_path = artifact_manifest_path(sam_path)
    corpus_id = Path(corpus_path).name
    model_slug = Path(model_name).name or model_name.replace("/", "_")
    manifest = build_artifact_manifest(
        artifact_id=f"{corpus_id}__{model_slug}__{text_response_field}",
        corpus_id=corpus_id,
        model_name=model_name,
        tokenizer_name_or_path=normalize_text(getattr(tokenizer, "name_or_path", "")) or model_name,
        response_field=text_response_field,
        cutoff_len=cutoff_len,
        sample_count=len(rows),
        effective_sample_count=len(effective_rows),
        skipped_count=skipped_count,
        truncated_count=truncated_count,
        domain_counts=count_by(effective_rows, "domain"),
        source_counts=count_by(effective_rows, "source_id"),
        strict_no_eval_dataset=validation["status"] == "passed",
        sam_path=str(sam_path),
        eos_token_id=int(eos_token_id),
        tokenizer_vocab_size=vocab_size,
        smoke_load=smoke_load,
    )
    try:
        load_sam_fn(str(sam_path))
        smoke_load = True
        manifest["smoke_load"] = True
    finally:
        save_json(manifest_path, manifest)

    print(f"built Static SAM artifact: {sam_path}")
    print(f"effective samples: {len(effective_rows)}")
    print(f"skipped samples: {skipped_count}")
    print(f"truncated samples: {truncated_count}")
    print(f"artifact manifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--text-response-field", choices=sorted(RESPONSE_FIELDS), required=True)
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--on-too-long", choices=("skip", "truncate"), default="skip")
    parser.add_argument("--sam-path", required=True)
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--allow-unregistered-local-source", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_artifact(
        corpus_path=args.corpus_path,
        model_name=args.model_name,
        text_response_field=args.text_response_field,
        cutoff_len=args.cutoff_len,
        sam_path=args.sam_path,
        registry_path=args.registry_path,
        on_too_long=args.on_too_long,
        allow_unregistered_local_source=args.allow_unregistered_local_source,
    )


if __name__ == "__main__":
    main()
