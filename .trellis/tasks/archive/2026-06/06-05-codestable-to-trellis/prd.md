# Migrate CodeStable to Trellis

## Goal

Migrate the durable project knowledge that was stored under `.codestable/` into Trellis so future Trellis implementation and check phases load the real SAM-Decoding constraints instead of generic placeholder specs.

## What I Already Know

* The repository already has Trellis initialized with `.trellis/spec/backend/`, `.trellis/spec/frontend/`, and `.trellis/spec/guides/`.
* The current Trellis backend specs are mostly placeholders.
* Before deletion, `.codestable/` contained the useful project memory: architecture, requirements, long-term decisions, learnings, and feature history.
* `AGENTS.md` already points assistants at Trellis, so the main missing piece is moving project knowledge into Trellis specs.
* This is a Python research codebase centered on SAM-Decoding, EAGLE/EAGLE2/EAGLE3 draft models, evaluation scripts, and Static SAM tooling. There is no frontend surface to migrate.

## Requirements

* Convert long-lived CodeStable knowledge into Trellis backend specs:
  * project/module architecture
  * dependency and runtime constraints
  * EAGLE3 integration rules
  * SAM + EAGLE3 tree-fusion rules
  * evaluation and V_miss benchmark protocols
  * Static SAM corpus/tooling requirements
  * testing and quality expectations
* Remove the source `.codestable/` directory after migration, per user request on 2026-06-05.
* Do not attempt a lossy one-to-one conversion of historical CodeStable feature/checklist/acceptance files into Trellis tasks.
* Add a migration inventory under this task so future maintainers know what was migrated before deletion.
* Update Trellis indexes so future agents can discover and load the migrated specs.
* Configure this task's Trellis context manifests to include the specs/research needed for implementation and checking.

## Acceptance Criteria

* [x] `.trellis/spec/backend/index.md` no longer reads as a generic template and links to migrated project-specific specs.
* [x] Backend specs contain the hard constraints from `.codestable/attention.md`, `.codestable/architecture/ARCHITECTURE.md`, and active `.codestable/compound/` decisions/learnings.
* [x] Requirement-level knowledge from `.codestable/requirements/` is represented in Trellis specs or a task research artifact.
* [x] `.trellis/tasks/06-05-codestable-to-trellis/research/codestable-inventory.md` records the migration mapping.
* [x] `implement.jsonl` and `check.jsonl` contain real Trellis context entries, not only seed examples.
* [x] `.codestable/` is removed after the durable knowledge has been migrated into Trellis.

## Definition of Done

* Specs/docs updated with project-specific content.
* Trellis context manifests updated.
* Syntax/structure checks run where available.
* Git dirty state reviewed so this task's changes are distinguishable from pre-existing user WIP.

## Technical Approach

Use a spec-first migration:

1. Inventory CodeStable sources and classify them by Trellis destination.
2. Write durable, implementation-relevant knowledge into `.trellis/spec/backend/`.
3. Remove `.codestable/` after user confirmation.
4. Add this task's migration inventory as traceability for the removed source tree.

## Decision (ADR-lite)

**Context**: CodeStable and Trellis organize knowledge differently. CodeStable has separate requirements, architecture, feature specs, and compound decisions; Trellis expects project conventions and contracts in `.trellis/spec/`, while task-specific work lives in `.trellis/tasks/`.

**Decision**: Migrate long-lived project facts into Trellis specs, then remove the old `.codestable/` directory after explicit user confirmation.

**Consequences**: Future Trellis tasks get the important project constraints automatically. Historical CodeStable source paths no longer exist in the working tree; the Trellis migration inventory records the source-to-destination mapping.

## Out of Scope

* Removing CodeStable skills from `.agents/skills/`.
* Rewriting Trellis workflow mechanics.
* Converting every historical CodeStable feature/checklist into a Trellis task.
* Changing SAM-Decoding runtime behavior.

## Technical Notes

* Trellis local architecture reference says project-specific rules belong in `.trellis/spec/`.
* Current `.trellis/config.yaml` is single-repo mode with backend/frontend layers.
* `.trellis/tasks/00-bootstrap-guidelines/prd.md` confirms the original bootstrap goal was to fill real project specs.
* Source inventory is persisted in `research/codestable-inventory.md`.
* Validation run: `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-05-codestable-to-trellis` passed with 8 implement entries and 6 check entries.
* Placeholder scan over backend specs and this task returned no template-marker matches before this note was added.
* 2026-06-05 update: user explicitly requested deleting the old CodeStable content after migration.
* Post-deletion validation: `.codestable/` no longer exists, no `.codestable` path references remain, and `task.py validate` still passes.
