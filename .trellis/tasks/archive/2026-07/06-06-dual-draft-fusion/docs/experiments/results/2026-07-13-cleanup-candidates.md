# Dual Draft Fusion — Cleanup Candidate List

**Date**: 2026-07-13
**Scope**: identify only; **do not delete** until reviewed.
**Branch**: `feature/dual-draft-fusion` (HEAD `f9cb328`)

This list is ordered by risk: safer items first. Anything with
"review before delete" is still in active or historical use and
should not be removed without an explicit decision.

---

## A. Safe to delete (tracked junk / local residuals)

These are either tracked accidental backups, OS noise, or already-
gitignored build/experiment residues. Deleting them does not change
any research conclusion.

| # | Path | Why | Size / count | Tracked? |
| -: | --- | --- | --- | --- |
| A1 | `evaluation/inference_sam_only.py.bak` | Stale side-by-side copy of live file (differs from live; date 2026-05-23). No code imports it. | ~5 KB | **yes** (tracked) |
| A2 | `samd/cache.py.bak` | Same pattern. Live `samd/cache.py` exists and differs. | ~4 KB | **yes** |
| A3 | `samd/sam/sam.py.bak` | Live file is now `static_sam.py` / `dyn_sam.py`; this is a leftover rename. | ~9 KB | **yes** |
| A4 | `evaluation/model/sam_only/cache.py.bak` | Same as A2 under the evaluation mirror. | ~4 KB | **yes** |
| A5 | `evaluation/model/sam_only/sam/sam.py.bak` | Same as A3 under the evaluation mirror. | ~9 KB | **yes** |
| A6 | 18× `.DS_Store` under `.`, `.cursor/`, `.agents/`, `.claude/`, `.codex/`, `tools/`, `evaluation/`, `evaluation/data/` | macOS Finder metadata. None of them are research artifacts. | small | **no** (untracked; also not in `.gitignore` currently — worth adding) |
| A7 | 33× `__pycache__/` trees | Python bytecode caches. | small | **no** |
| A8 | `.worktrees/2026-06-11-drafter-mars-sam-gate/` | Local residual of the experiment worktree (gitignored via `.worktrees/`). The corresponding `git worktree` entry is already **prunable** and points at a deleted path under `paper_code/`. | **4.8 MB** | **no** (gitignored) |
| A9 | `git worktree prune` for the dead entry | `git worktree list` shows `.../paper_code/.../2026-06-11-drafter-mars-sam-gate` as **prunable**. Run `git worktree prune` to drop the registry entry. | n/a | registry only |

Suggested first batch (lowest risk):

```bash
# tracked bak files — will show up in git status after deletion
git rm \
  evaluation/inference_sam_only.py.bak \
  samd/cache.py.bak \
  samd/sam/sam.py.bak \
  evaluation/model/sam_only/cache.py.bak \
  evaluation/model/sam_only/sam/sam.py.bak

# untracked OS / cache noise
find . -name '.DS_Store' -not -path './.git/*' -delete
find . -name '__pycache__' -type d -not -path './.git/*' -exec rm -rf {} +

# worktree residual (gitignored) + prune registry
rm -rf .worktrees/2026-06-11-drafter-mars-sam-gate
git worktree prune

# optional: stop future .DS_Store noise
# echo '.DS_Store' >> .gitignore
```

---

## B. Large gitignored data (local disk only; not in git)

`evaluation/data` is entirely gitignored. Deleting any of these frees
local disk but has **no effect on the repository**. Keep anything you
still want to re-analyze offline.

| # | Path | Why | Size | Recommendation |
| -: | --- | --- | ---: | --- |
| B1 | `evaluation/data/medquad/` | Closed OOV / linear-alignment probe dumps (2026-05-28 era). No active experiment references this for Drafter-MARS. | **115 MB** | Free if you do not plan to re-run OOV analysis. |
| B2 | `evaluation/data/mt_bench/` | Profile traces for MT-Bench naive fusion / overhead. MT-Bench is out of scope for Drafter-MARS (ceiling too low). | **12 MB** | Free if MT-Bench offline analysis is done. |
| B3 | `evaluation/data/medqa/` | MedQA model answers + traces. MedQA is out of scope for Drafter-MARS. | **3.4 MB** | Free if MedQA offline analysis is done. |
| B4 | `evaluation/data/humaneval/drafter_mars_sam_gate/` | Currently only holds `drafter_mars_smoke_q0_20.provenance.txt` (792 B). Keep — this is the only local smoke provenance. | tiny | **keep** |
| B5 | `evaluation/data/humaneval/question.jsonl` / `question_canonical.jsonl` | Question files. `question_canonical.jsonl` is the SHA256 `fc49f930...` baseline; the smoke provenance used the older `d49a763b...` file. | small | **keep both** until smoke is re-run on the canonical file. |

---

## C. Possibly obsolete source (review before delete)

These are still present in the repo and may be useful for other
experiments. They are **not** required by the Drafter-MARS SAM-gate
path, but deleting them is a design decision, not a cleanup of junk.

| # | Path | Why candidate | Refs outside itself | Recommendation |
| -: | --- | --- | ---: | --- |
| C1 | `evaluation/analyze_linear_alignment.py` | Closed 2026-05-28 OOV / CCA probe analyzer. **0 code/script references** outside itself; only mentioned in archived closed-experiment plans. | 0 (active) | Strongest candidate among live sources. Safe to archive or delete after you confirm you will not re-run OOV analysis. |
| C2 | `evaluation/analyze_vmiss.py` | Vocabulary-miss analyzer from earlier probes. Still referenced by scripts (`run_medqa_vmiss_eval.sh`, `test_medqa_vmiss_smoke.sh`). | ~7 | Keep until MedQA vmiss scripts are retired. |
| C3 | `evaluation/oracle_depth_decoupled.py` | Alternative oracle used in the canonical baseline table (`depth_decoupled` +1.16%). Still referenced. | ~7 | **keep** — baseline comparison depends on it. |
| C4 | `evaluation/oracle_high_precision_sam.py` | High-precision SAM oracle variant. Some refs remain. | ~4 | Review; not used by Drafter-MARS path. |
| C5 | `evaluation/oracle_rejection_boundary.py` | Rejection-boundary oracle utility. Referenced by scripts (`eval_rejection_boundary_*.sh`). | ~7 | **keep** — used to produce the +2.72% ceiling. |
| C6 | `evaluation/model/sam_only/` | Standalone SAM-only evaluation mirror of `samd`. Used by `inference_sam_only.py` and related scripts. | active scripts | **keep** unless you retire the SAM-only baseline. |
| C7 | `.trellis/tasks/06-06-dual-draft-fusion/scripts/run_mtbench_current_compare_20260608.sh` | One-off MT-Bench compare from 2026-06-08. MT-Bench is out of scope. | 0 | Safe to archive under `docs/archive/` or delete. |
| C8 | `scripts/eval_boundary_graft*.sh`, `scripts/eval_rejection_boundary_*.sh`, `scripts/run_medqa_*`, `scripts/run_medquad_*`, `scripts/profile_fusion_medqa.sh`, `scripts/profile_fusion_mt_bench.sh` | Scripts for closed or out-of-scope experiments (boundary-graft online path is blocked; MT-Bench/MedQA ceilings too low). | vary | Do not bulk-delete. Mark which ones are still needed for Stage 2 / other papers, archive the rest. |
| C9 | `.trellis/tasks/06-06-dual-draft-fusion/docs/archive/` | Historical PRDs, phase notes, closed experiments, old research. Already archived (492 KB). | n/a | **keep** — this is the intended resting place for closed work. Do not re-delete. |
| C10 | `MARS/` (3.3 MB) | Vendor reference for the MARS ratio signal. PRD and predictor module both cite it. | active | **keep**. |

---

## D. Stale documentation state (not files to delete — to update)

These are not candidates for deletion; they are **out of date** relative
to the code and should be rewritten when you do the doc pass:

| # | Path | Stale claim | Reality |
| -: | --- | --- | --- |
| D1 | `.trellis/tasks/06-06-dual-draft-fusion/README.md` | "Drafter-MARS has never been implemented or tested. … `analyze_boundary_predictor.py` currently only supports `low_margin`." Implementation checklist all unchecked. | Module + suite + tests exist and are committed. |
| D2 | `docs/experiments/README.md` | Drafter-MARS status = "Implementation pending". | Implementation is done; smoke numbers are the open item. |
| D3 | `docs/experiments/results/` | Missing the smoke decision note required by the plan (`2026-06-11-drafter-mars-sam-gate.md`). | Only baselines and this session's report exist. |
| D4 | Smoke provenance | Uses question file SHA256 `d49a763b...` (old q0-20). | Canonical baseline is `fc49f930...`. Re-run smoke on the canonical file. |

---

## E. Recommended order of operations

1. **A batch** (tracked `.bak` + `.DS_Store` + `__pycache__` + worktree residual + `git worktree prune`). One small commit: "Remove stale bak files and local residuals."
2. Optionally add `.DS_Store` to `.gitignore` in the same commit.
3. **B batch** only if you want the disk back; no git impact.
4. **C1 / C7** after a one-line confirmation they will not be re-run.
5. **D** is a documentation rewrite, not a delete — do it after the smoke re-run so the result note and the README stay consistent.

Nothing in this list is a source-code refactor. The Drafter-MARS
implementation (`evaluation/drafter_mars_predictor.py`,
`analyze_boundary_predictor.py`, `samd/tree_model/eagle3/*`,
`samd/fusion/naive_fusion.py`, `samd/utils.py`,
`evaluation/oracle_fusion_analysis.py`) is intentionally **out of
scope** for cleanup.
