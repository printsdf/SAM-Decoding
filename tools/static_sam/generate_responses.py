"""Generate model-style responses for a Static SAM canonical corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .corpus_io import load_rows, save_manifest, save_rows
from .registry import (
    DEFAULT_REGISTRY_PATH,
    find_allowed_source,
    load_registry,
)
from .schema import build_corpus_manifest, has_text, now_iso, normalize_bool, normalize_text
from .validate_corpus import validate_rows


ResponseProvider = Callable[[list[str]], list[str]]


def _ensure_generation_allowed(rows: list[dict[str, Any]], registry: dict[str, Any]) -> None:
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        source_id = normalize_text(row.get("source_id"))
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        allowed = find_allowed_source(registry, source_id)
        if allowed is not None and not normalize_bool(allowed.get("allow_generated_response", True)):
            errors.append(f"source_id {source_id!r} has allow_generated_response=false")
    if errors:
        raise ValueError("; ".join(errors))


def _vllm_provider(
    *,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> ResponseProvider:
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "generate_responses requires `vllm` unless a response_provider is supplied"
        ) from exc

    llm = LLM(model=model_name, enable_prefix_caching=True)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    def generate(prompts: list[str]) -> list[str]:
        outputs = llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    return generate


def generate_responses(
    *,
    corpus_path: str | Path,
    model_name: str,
    registry_path: str | Path | None = None,
    output_field: str = "generated_response",
    limit: int | None = None,
    resume: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 1024,
    response_provider: ResponseProvider | None = None,
) -> dict[str, Any]:
    if output_field != "generated_response":
        raise ValueError("Only output_field='generated_response' is supported")

    rows = load_rows(corpus_path)
    registry = load_registry(registry_path or DEFAULT_REGISTRY_PATH)
    report = validate_rows(rows, registry)
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    _ensure_generation_allowed(rows, registry)

    candidate_indices: list[int] = []
    skipped_existing = 0
    for idx, row in enumerate(rows):
        if resume and has_text(row.get(output_field)):
            skipped_existing += 1
            continue
        candidate_indices.append(idx)

    if limit is not None:
        candidate_indices = candidate_indices[:limit]

    prompts = [normalize_text(rows[idx].get("prompt")) for idx in candidate_indices]
    if response_provider is None and prompts:
        response_provider = _vllm_provider(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    responses = response_provider(prompts) if prompts and response_provider is not None else []
    if len(responses) != len(prompts):
        raise ValueError(f"response provider returned {len(responses)} responses for {len(prompts)} prompts")

    generated_at = now_iso()
    for idx, response in zip(candidate_indices, responses):
        rows[idx][output_field] = normalize_text(response)
        rows[idx]["response_generator"] = model_name
        rows[idx]["response_generated_at"] = generated_at

    generation = {
        "model_name": model_name,
        "output_field": output_field,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "limit": limit,
        "resume": resume,
        "generated_count": len(responses),
        "skipped_existing_count": skipped_existing,
        "generated_at": generated_at if responses else "",
    }
    validation = validate_rows(rows, registry)
    save_rows(corpus_path, rows)
    save_manifest(
        corpus_path,
        build_corpus_manifest(corpus_path, rows, validation=validation, generation=generation),
    )
    print(f"generated responses: {len(responses)}")
    print(f"skipped existing responses: {skipped_existing}")
    return generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--output-field", default="generated_response")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_responses(
        corpus_path=args.corpus_path,
        model_name=args.model_name,
        registry_path=args.registry_path,
        output_field=args.output_field,
        limit=args.limit,
        resume=not args.overwrite_existing,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
