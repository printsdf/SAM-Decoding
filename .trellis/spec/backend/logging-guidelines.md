# Logging Guidelines

> Script output, progress notifications, and diagnostic records.

## Overview

This project mainly uses stdout/stderr and shell-script notifications rather
than a logging framework. Keep output concise but sufficient to reproduce what
ran, which model/config was used, and where artifacts were written.

## What to Print

* CLI inference entry points should print important runtime choices such as
  `tree_method`, `tree_fusion`, `tree_model_path`, `max_cache_len`, and output
  paths.
* Benchmark scripts should print phase boundaries, elapsed time, bench name,
  group/model id, answer file path, and analysis output.
* Analysis scripts should print warnings for skipped malformed trace steps and
  for answer files that lack `diagnosis_traces`.
* Fusion code may print compact aggregate stats at the end of generation when
  the mode collects useful counters.

## Batch Script Notifications

For long benchmark batches, follow the pattern in
`scripts/run_bench_cross_domain_speedup.sh`:

* One top-level shell script should run fetch, phase1 inference, phase1
  analysis, phase2 inference, and phase2 analysis in order.
* Notification messages should mark stage progress and failures clearly.
* Do not split a single benchmark workflow into multiple sub-scripts that require
  repeated user triggering unless there is a concrete operational reason.

## What Not to Log

* Do not print credentials, private webhook URLs, access tokens, or full
  environment dumps.
* Avoid dumping full generated answers in batch scripts; write them to answer
  files and print paths/summaries.
* Do not print large per-step diagnosis traces to stdout in normal benchmark
  runs; persist them in JSONL answer files.

## Log Level Convention

There is no structured log level system. Use clear text prefixes in scripts:

* `START` / `DONE` for stage boundaries.
* `WARNING` for recoverable malformed inputs or suspicious but non-fatal states.
* `ERROR` or shell `trap ERR` handling for command failures.
