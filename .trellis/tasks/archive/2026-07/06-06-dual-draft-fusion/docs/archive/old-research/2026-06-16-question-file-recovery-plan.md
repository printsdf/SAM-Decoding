# HumanEval Question File Recovery Plan

Created: 2026-06-16
Task: `dual-draft-fusion`
Context: Root cause diagnosis found that q0-20 smoke low ceiling is caused by
question file provenance mismatch.

## Problem

Three different HumanEval question files exist across traces:

| Trace | File rows | SHA256 (prefix) | EAGLE MAT |
|-------|-----------|-----------------|-----------|
| Original q0-164 oracle | ≥164 | unknown (lost) | 3.67 |
| q0-20 Drafter-MARS smoke | 20 | `d49a763b` | 6.79 |
| Current local question.jsonl | 80 | `ce0244e2` | — |

The original 164-question file that produced the validated `+9.65%` oracle
evidence is lost. The q0-20 smoke used a trivial 20-row subset that EAGLE
handles too well.

## Recovery Options

### Option 1: Reconstruct from HumanEval standard benchmark

Source: https://github.com/openai/human-eval

Standard HumanEval contains 164 programming problems. The canonical format is:

```python
{
    "task_id": "HumanEval/0",
    "prompt": "...",
    "canonical_solution": "...",
    "test": "...",
    "entry_point": "..."
}
```

Current project format (from local 80-row file):

```json
{
    "question_id": 0,
    "category": "code",
    "turns": ["Complete the code I provided.\n\n<prompt>"],
    "reference": ["<solution>"]
}
```

**Action**: Write a conversion script that:
1. Downloads/loads HumanEval 164 problems
2. Transforms to project format
3. Writes `evaluation/data/humaneval/question.jsonl` (164 rows)
4. Records SHA256 in provenance

### Option 2: Recover from model host backup

The original oracle trace was run on `/root/SAM-Decoding/...` (remote model
environment). If that checkout still exists, copy the original question file.

**Risk**: The remote environment may have been cleaned up or the file may no
longer match any git-tracked version.

### Option 3: Use current 80-row file and validate

Accept the current 80-row local file as the new baseline. Run a full q0-80
oracle profile and verify EAGLE MAT ≈ 3.5–4.0 and oracle ceiling ≥ +7%.

**Risk**: Only 80 questions may not provide enough calibration data for a
stable Drafter-MARS predictor.

## Recommendation

**Use Option 1** (reconstruct from standard HumanEval):

- Canonical source, reproducible
- 164 questions match the original validated oracle evidence
- Can pin SHA256 for all future traces
- Low risk: standard benchmark is well-defined

## Implementation Checklist

- [ ] Write `evaluation/humaneval_prep.py` to convert HumanEval format to
      project format
- [ ] Download HumanEval 164 problems (from `human_eval` package or GitHub)
- [ ] Generate `evaluation/data/humaneval/question.jsonl` (164 rows)
- [ ] Record SHA256 in a `.provenance` file
- [ ] Run q0-20 smoke with new file, verify EAGLE MAT ≈ 3.5–4.0
- [ ] Verify same-trace oracle ceiling ≥ +7%
- [ ] If ceiling confirmed, proceed to Drafter-MARS full calibration

## Next Session

Start with Option 1 implementation. Write the prep script first, then generate
the canonical 164-row question file.
