"""Smoke tests for the Static SAM offline corpus pipeline.

The tests use fake tokenization/SAM builders so they do not require torch,
transformers, vLLM, or model weights.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.static_sam.build_artifact import build_artifact  # noqa: E402
from tools.static_sam.corpus_io import load_json, load_rows  # noqa: E402
from tools.static_sam.generate_responses import generate_responses  # noqa: E402
from tools.static_sam.import_local import import_local  # noqa: E402
from tools.static_sam.validate_corpus import validate_corpus  # noqa: E402


CANONICAL_FIELDS = {
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
}


class FakeTokenizer:
    eos_token_id = 2
    name_or_path = "fake-tokenizer"

    def __len__(self) -> int:
        return 5

    def __call__(self, text: str, padding: bool = False, return_tensors: Any = None) -> dict[str, list[int]]:
        return {"input_ids": [(ord(ch) % 50) + 3 for ch in text]}


def fake_build_sam(batch_tokens: list[list[int]], eos_token_id: int) -> dict[str, Any]:
    return {"batch_tokens": batch_tokens, "eos_token_id": eos_token_id}


def fake_dump_sam(path: str, sam: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(sam), encoding="utf-8")


def fake_load_sam(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def import_fixture_corpus(root: Path) -> Path:
    medical = root / "medical.jsonl"
    finance = root / "finance.jsonl"
    write_jsonl(
        medical,
        [
            {"question": "q", "context": "", "answer": "a"},
            {"question": "long", "context": "", "answer": "answer"},
        ],
    )
    write_jsonl(
        finance,
        [
            {"question": "APR?", "context": "", "answer": "Annual percentage rate."},
            {"question": "Revenue?", "context": "", "answer": "Income from sales."},
        ],
    )
    corpus = root / "corpus"
    import_local(
        input_path=medical,
        output_path=corpus,
        domain="medical",
        source_id="local_medical_qa_v1",
        source_name="",
        license="",
        eval_exclusion_reason="",
        instruction_field="question",
        input_field="context",
        response_field="answer",
        prompt_template_name="none",
    )
    import_local(
        input_path=finance,
        output_path=corpus,
        domain="finance",
        source_id="local_finance_qa_v1",
        source_name="",
        license="",
        eval_exclusion_reason="",
        instruction_field="question",
        input_field="context",
        response_field="answer",
        prompt_template_name="none",
    )
    return corpus


def assert_system_exit(fn, message: str) -> None:
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError(message)


def run_pipeline_smoke(root: Path) -> None:
    corpus = import_fixture_corpus(root)
    rows = load_rows(corpus)
    manifest = load_json(corpus / "manifest.json")
    assert len(rows) == 4
    assert CANONICAL_FIELDS.issubset(rows[0].keys())
    assert manifest["domains"] == {"finance": 2, "medical": 2}
    assert manifest["sources"]["local_medical_qa_v1"]["count"] == 2
    assert manifest["sources"]["local_finance_qa_v1"]["count"] == 2

    generation = generate_responses(
        corpus_path=corpus,
        model_name="stub-model",
        limit=3,
        response_provider=lambda prompts: [f"generated: {prompt}" for prompt in prompts],
    )
    assert generation["generated_count"] == 3
    generation = generate_responses(
        corpus_path=corpus,
        model_name="stub-model",
        response_provider=lambda prompts: [f"generated: {prompt}" for prompt in prompts],
    )
    assert generation["skipped_existing_count"] == 3
    assert generation["generated_count"] == 1

    for field in ("source_response", "generated_response"):
        sam_path = root / f"{field}.pkl"
        artifact_manifest = build_artifact(
            corpus_path=corpus,
            model_name="fake-model",
            text_response_field=field,
            cutoff_len=100,
            sam_path=sam_path,
            tokenizer=FakeTokenizer(),
            build_sam_fn=fake_build_sam,
            dump_sam_fn=fake_dump_sam,
            load_sam_fn=fake_load_sam,
        )
        disk_manifest = load_json(sam_path.with_suffix(".manifest.json"))
        assert sam_path.exists()
        assert artifact_manifest["response_field"] == field
        assert disk_manifest["response_field"] == field
        assert disk_manifest["smoke_load"] is True
        assert disk_manifest["strict_no_eval_dataset"] is True
        assert disk_manifest["domain_counts"] == {"finance": 2, "medical": 2}
        assert disk_manifest["tokenizer_vocab_size"] == 5


def run_guardrail_smoke(root: Path) -> None:
    denied = root / "denied.jsonl"
    eval_data = root / "eval.jsonl"
    unregistered = root / "unregistered.jsonl"
    write_jsonl(denied, [{"question": "q", "context": "", "answer": "a"}])
    write_jsonl(eval_data, [{"question": "q", "context": "", "answer": "a"}])
    write_jsonl(unregistered, [{"question": "q", "context": "", "answer": "a"}])

    assert_system_exit(
        lambda: import_local(
            input_path=denied,
            output_path=root / "denied_corpus",
            domain="medical",
            source_id="medqa",
            source_name="MedQA",
            license="unknown",
            eval_exclusion_reason="benchmark",
            instruction_field="question",
            input_field="context",
            response_field="answer",
            prompt_template_name="none",
        ),
        "denied source should fail",
    )

    assert_system_exit(
        lambda: import_local(
            input_path=eval_data,
            output_path=root / "eval_corpus",
            domain="medical",
            source_id="local_medical_qa_v1",
            source_name="",
            license="",
            eval_exclusion_reason="",
            instruction_field="question",
            input_field="context",
            response_field="answer",
            prompt_template_name="none",
            is_eval_dataset=True,
        ),
        "is_eval_dataset=True should fail",
    )

    assert_system_exit(
        lambda: import_local(
            input_path=unregistered,
            output_path=root / "unknown_corpus",
            domain="medical",
            source_id="unknown_source",
            source_name="Unknown",
            license="internal",
            eval_exclusion_reason="local non-benchmark data",
            instruction_field="question",
            input_field="context",
            response_field="answer",
            prompt_template_name="none",
        ),
        "unregistered source should fail by default",
    )

    allowed_corpus = root / "unknown_allowed_corpus"
    import_local(
        input_path=unregistered,
        output_path=allowed_corpus,
        domain="medical",
        source_id="unknown_source",
        source_name="Unknown",
        license="internal",
        eval_exclusion_reason="local non-benchmark data",
        instruction_field="question",
        input_field="context",
        response_field="answer",
        prompt_template_name="none",
        allow_unregistered_local_source=True,
    )
    report = validate_corpus(allowed_corpus, allow_unregistered_local_source=True)
    assert report["unregistered_sources"] == ["unknown_source"]


def run_response_and_length_smoke(root: Path) -> None:
    corpus = import_fixture_corpus(root)
    assert_system_exit(
        lambda: build_artifact(
            corpus_path=corpus,
            model_name="fake-model",
            text_response_field="generated_response",
            cutoff_len=100,
            sam_path=root / "missing_generated.pkl",
            tokenizer=FakeTokenizer(),
            build_sam_fn=fake_build_sam,
            dump_sam_fn=fake_dump_sam,
            load_sam_fn=fake_load_sam,
        ),
        "empty generated_response should fail",
    )

    skipped_manifest = build_artifact(
        corpus_path=corpus,
        model_name="fake-model",
        text_response_field="source_response",
        cutoff_len=3,
        sam_path=root / "skip.pkl",
        tokenizer=FakeTokenizer(),
        build_sam_fn=fake_build_sam,
        dump_sam_fn=fake_dump_sam,
        load_sam_fn=fake_load_sam,
    )
    assert skipped_manifest["skipped_count"] > 0
    assert skipped_manifest["truncated_count"] == 0

    truncated_manifest = build_artifact(
        corpus_path=corpus,
        model_name="fake-model",
        text_response_field="source_response",
        cutoff_len=3,
        on_too_long="truncate",
        sam_path=root / "truncate.pkl",
        tokenizer=FakeTokenizer(),
        build_sam_fn=fake_build_sam,
        dump_sam_fn=fake_dump_sam,
        load_sam_fn=fake_load_sam,
    )
    assert truncated_manifest["skipped_count"] == 0
    assert truncated_manifest["truncated_count"] > 0


def test_static_sam_pipeline(tmp_path):
    run_pipeline_smoke(tmp_path / "pipeline")
    run_guardrail_smoke(tmp_path / "guardrails")
    run_response_and_length_smoke(tmp_path / "length")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_pipeline_smoke(root / "pipeline")
        run_guardrail_smoke(root / "guardrails")
        run_response_and_length_smoke(root / "length")
    print("static SAM pipeline smoke PASS")


if __name__ == "__main__":
    main()
