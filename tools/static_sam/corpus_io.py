"""Filesystem helpers for Static SAM canonical corpora."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def require_datasets():
    try:
        from datasets import Dataset, load_from_disk  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Static SAM corpus tooling requires `datasets`: pip install datasets"
        ) from exc
    return Dataset, load_from_disk


def dataset_path(corpus_path: str | Path) -> Path:
    return Path(corpus_path) / "dataset"


def manifest_path(corpus_path: str | Path) -> Path:
    return Path(corpus_path) / "manifest.json"


def load_rows(corpus_path: str | Path) -> list[dict[str, Any]]:
    _, load_from_disk = require_datasets()
    path = dataset_path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Canonical dataset not found: {path}")
    dataset = load_from_disk(str(path))
    return [dict(row) for row in dataset]


def save_rows(corpus_path: str | Path, rows: list[dict[str, Any]]) -> None:
    Dataset, _ = require_datasets()
    root = Path(corpus_path)
    root.mkdir(parents=True, exist_ok=True)
    path = dataset_path(root)
    tmp_path = root / "dataset.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    Dataset.from_list(rows).save_to_disk(str(tmp_path))
    if path.exists():
        shutil.rmtree(path)
    tmp_path.rename(path)


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def save_manifest(corpus_path: str | Path, manifest: dict[str, Any]) -> None:
    save_json(manifest_path(corpus_path), manifest)


def copy_registry_snapshot(registry_path: str | Path, corpus_path: str | Path) -> None:
    source = Path(registry_path)
    if not source.exists():
        return
    target = Path(corpus_path) / "data_sources.lock.yaml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
