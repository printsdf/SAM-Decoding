# Backend Development Guidelines

> Project-specific engineering rules for SAM-Decoding.

This repository is a Python research codebase for suffix-automaton-based
speculative decoding. Most work touches model integration code, evaluation
scripts, benchmark data preparation, or Static SAM tooling. There is no backend
service, database, or web API layer.

## Pre-Development Checklist

Before changing backend code, read the files that match the area you will touch:

| Area | Read First |
| --- | --- |
| Any backend change | [Directory Structure](./directory-structure.md), [Quality Guidelines](./quality-guidelines.md), [Runtime Constraints](./runtime-constraints.md) |
| SAMD core or model patches | [Project Architecture](./project-architecture.md), [EAGLE3 Integration](./eagle3-integration.md), [Error Handling](./error-handling.md) |
| Tree fusion or candidate generation | [Tree Fusion](./tree-fusion.md), [EAGLE3 Integration](./eagle3-integration.md) |
| Evaluation, benchmark, or diagnosis trace | [Evaluation Protocols](./evaluation-protocols.md), [Logging Guidelines](./logging-guidelines.md) |
| Static SAM data/artifact tooling | [Project Capabilities](./project-capabilities.md), [Database Guidelines](./database-guidelines.md) |

Also read the shared thinking guides when the change spans multiple modules or
duplicates an existing pattern:

* [Code Reuse Thinking Guide](../guides/code-reuse-thinking-guide.md)
* [Cross-Layer Thinking Guide](../guides/cross-layer-thinking-guide.md)

## Guidelines Index

| Guide | Description | Status |
| --- | --- | --- |
| [Project Architecture](./project-architecture.md) | Current module map and responsibility boundaries | Current |
| [Runtime Constraints](./runtime-constraints.md) | Dependencies, model/runtime limits, tokenizer and cache constraints | Current |
| [EAGLE3 Integration](./eagle3-integration.md) | EAGLE3 draft-model contracts and known pitfalls | Current |
| [Tree Fusion](./tree-fusion.md) | SAM + EAGLE3 candidate-tree fusion contracts | Current |
| [Evaluation Protocols](./evaluation-protocols.md) | V_miss, trace, speedup, and benchmark rules | Current |
| [Project Capabilities](./project-capabilities.md) | Current and draft capability intent migrated from CodeStable requirements | Current |
| [Directory Structure](./directory-structure.md) | Module organization and file layout | Current |
| [Database Guidelines](./database-guidelines.md) | No database; artifact/data-file conventions | Current |
| [Error Handling](./error-handling.md) | Fail-fast, warning, and script error conventions | Current |
| [Quality Guidelines](./quality-guidelines.md) | Testing and review standards | Current |
| [Logging Guidelines](./logging-guidelines.md) | Script output and benchmark notification conventions | Current |

## Source of Truth

These specs are the active Trellis source of truth for future AI work.
The old `.codestable/` directory was removed after migration; use
`.trellis/tasks/06-05-codestable-to-trellis/research/codestable-inventory.md`
for source-to-destination traceability. New durable project rules should be
added under `.trellis/spec/`.
