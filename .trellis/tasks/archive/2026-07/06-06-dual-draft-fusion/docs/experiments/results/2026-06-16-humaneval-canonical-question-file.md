# HumanEval Canonical Question File Generation

Date: 2026-06-16
Task: `dual-draft-fusion`
Status: Baseline established

## Summary

Generated a canonical 164-question HumanEval question file from the official
OpenAI HumanEval benchmark to replace the inconsistent question files that
caused the q0-20 smoke low-ceiling confound.

## Motivation

Root cause analysis (2026-06-16) found that three different HumanEval question
files existed across experiments:

| Trace | File rows | SHA256 (prefix) | EAGLE MAT | Status |
|-------|-----------|-----------------|-----------|--------|
| Original q0-164 oracle | ≥164 | unknown | 3.67 | Lost |
| q0-20 Drafter-MARS smoke | 20 | `d49a763b` | 6.79 | Trivial subset |
| Previous local | 80 | `ce0244e2` | — | Non-standard |

The q0-20 smoke used a standalone 20-row file of trivial function-completion
problems, not a subset of the original benchmark. EAGLE handled these too well
(MAT 6.79 vs expected 3.67), leaving almost no SAM rescue headroom and
producing a misleading `+3.91%` oracle ceiling.

## Implementation

Created `evaluation/humaneval_prep.py` to materialize the canonical benchmark:

```bash
python evaluation/humaneval_prep.py \
    --output evaluation/data/humaneval/question.jsonl
```

### Source

- Primary: `human_eval` Python package (`pip install human-eval`)
- Fallback: Direct download from https://github.com/openai/human-eval

### Output Format

Project-standard question format (consistent with MedQA, MT-Bench):

```json
{
    "question_id": 0,
    "category": "code",
    "turns": ["Complete the code I provided.\n\n<prompt>"],
    "reference": ["<canonical_solution>"]
}
```

## Generated File

- **Path**: `evaluation/data/humaneval/question.jsonl`
- **Rows**: 164 (complete HumanEval benchmark)
- **SHA256**: `fc49f9304222ac25c3ff5c4ede32e2d61646bd08307da80f0ad806ee60952407`
- **First problem**: `has_close_elements` (HumanEval/0)
- **Last problem**: `anti_shuffle` (HumanEval/163)

## Validation

- ✅ 164 rows (complete benchmark)
- ✅ JSON format valid (all lines parse)
- ✅ Field structure matches project schema
- ✅ Problem order matches official HumanEval task_id sequence
- ✅ SHA256 pinned for provenance tracking

## Next Step

Re-run q0-20 smoke with this canonical file to verify:

1. EAGLE-only MAT returns to ~3.5–4.0 (consistent with original oracle trace)
2. Same-trace perfect/rejection-boundary oracle ceiling ≥ +7%
3. If ceiling confirmed, proceed to Drafter-MARS full q0-164 calibration

Command for q0-20 smoke:

```bash
export QUESTION_BEGIN=0
export QUESTION_END=20
export TRAIN_BEGIN=0
export TRAIN_END=10
export VALID_BEGIN=10
export VALID_END=20
export MODEL_ID=drafter_mars_canonical_smoke_q0_20
export PROFILE_OUTPUT_DIR=evaluation/data/humaneval/canonical_smoke
export BOUNDARY_PREDICTOR_SUITE=drafter_mars

bash scripts/profile_boundary_predictor_humaneval.sh
```

Expected EAGLE MAT for q0-20 subset: ~3.5–4.0 (if consistent with full set)

## Files Updated

- `evaluation/humaneval_prep.py` — new prep script
- `evaluation/data/humaneval/question.jsonl` — canonical 164-question file
- This result note

## Avoid Repetition

- Do not use the old 20-row or 80-row question files for HumanEval experiments.
- Always verify question file SHA256 in experiment provenance.
- Pin this SHA256 (`fc49f930...`) as the canonical baseline for all future
  HumanEval traces.
