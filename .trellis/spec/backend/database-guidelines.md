# Database and Artifact Guidelines

> This project has no application database. The persistent data contracts are
> local corpora, model answers, benchmark outputs, and SAM artifacts.

## Overview

Do not introduce a database, ORM, or migration layer unless a future feature
explicitly requires one. Most durable data lives in:

* `evaluation/data/**` for benchmark questions and model answers.
* `tools/static_sam/**` for corpus schemas, registry entries, and artifact
  builders.
* `.trellis/tasks/<task>/docs/experiments/**` for human-readable experiment
  specs, plans, and results.

## Data and Artifact Patterns

* Keep benchmark question files in the Spec-Bench-compatible `question.jsonl`
  shape used by `evaluation/eval_llama3.py` and related runners.
* Keep model answers as JSONL files under each benchmark's `model_answer/`
  directory, with distinct `model_id` suffixes when multiple phases are run.
* When generating Static SAM artifacts, keep the corpus model-independent and
  bind artifacts to a tokenizer/model only at artifact-build time.
* Record source, domain, license/permission status, and evaluation-exclusion
  rationale for Static SAM corpora.

## Prohibited Patterns

* Do not mix evaluation benchmark data into Static SAM offline corpora, even if
  the dataset has a train split.
* Do not overwrite phase-specific answer files; use `_p1`, `_p2`, `_trace`, or
  similarly explicit model-id suffixes.
* Do not store opaque SAM artifacts without a reproducible source/corpus path
  and tokenizer/model provenance.

## Common Mistakes

* Treating benchmark JSONL as scratch output and overwriting it between
  trace-OFF and trace-ON runs.
* Building Static SAM from data whose relationship to the evaluation set is not
  documented.
* Binding corpus normalization to one model when it should remain model-neutral.
