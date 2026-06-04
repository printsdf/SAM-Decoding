"""Data source registry helpers for Static SAM corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("data_sources.yaml")


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip("'").strip('"')
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'").strip('"') for item in inner.split(",")]
    return value


def _limited_yaml_load(text: str) -> dict[str, Any]:
    """Parse the simple registry YAML shape without requiring PyYAML."""

    data: dict[str, Any] = {}
    current_key: str | None = None
    current_item: dict[str, Any] | None = None

    def finish_item() -> None:
        nonlocal current_item
        if current_key is not None and current_item is not None:
            data.setdefault(current_key, []).append(current_item)
        current_item = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        stripped = raw_line.strip()
        if not raw_line.startswith(" ") and stripped.endswith(":"):
            finish_item()
            current_key = stripped[:-1]
            data.setdefault(current_key, [])
            continue
        if stripped.startswith("- "):
            finish_item()
            current_item = {}
            remainder = stripped[2:].strip()
            if remainder and ":" in remainder:
                key, value = remainder.split(":", 1)
                current_item[key.strip()] = _parse_scalar(value)
            continue
        if current_item is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_item[key.strip()] = _parse_scalar(value)
    finish_item()
    return data


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    if not registry_path.exists():
        return {"allowed_sources": [], "denied_sources": []}
    text = registry_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
    except ImportError:
        # JSON is valid YAML and is handy in minimal environments.
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            loaded = _limited_yaml_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Registry {registry_path} must be a mapping")
    loaded.setdefault("allowed_sources", [])
    loaded.setdefault("denied_sources", [])
    return loaded


def _find_source(registry: dict[str, Any], group: str, source_id: str) -> dict[str, Any] | None:
    for entry in registry.get(group, []) or []:
        if str(entry.get("source_id", "")).lower() == source_id.lower():
            return entry
    return None


def find_allowed_source(registry: dict[str, Any], source_id: str) -> dict[str, Any] | None:
    return _find_source(registry, "allowed_sources", source_id)


def find_denied_source(registry: dict[str, Any], source_id: str) -> dict[str, Any] | None:
    return _find_source(registry, "denied_sources", source_id)


def allowed_domains(registry: dict[str, Any], defaults: set[str]) -> set[str]:
    domains = set(defaults)
    for entry in registry.get("allowed_sources", []) or []:
        domain = str(entry.get("domain", "")).strip()
        if domain:
            domains.add(domain)
    return domains
