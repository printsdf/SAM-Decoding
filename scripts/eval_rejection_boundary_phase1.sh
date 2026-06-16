#!/usr/bin/env bash
# Evaluate Phase 1 prototype on 20 HumanEval samples.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-/root/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B}
SAM_PATH=${SAM_PATH:-}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
DTYPE=${DTYPE:-float16}

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

# Create output directory
mkdir -p evaluation/data/humaneval/rejection_boundary_phase1

cmd=(
  "${PYTHON_BIN}" -m evaluation.inference_samd
  --template llama3
  --model-type llama3
  --model-id rejection_boundary_phase1
  --bench-name humaneval
  --question-begin 0
  --question-end 20
  --fusion_mode rejection_boundary
  --rejection_conf_threshold 0.5
  --model-path "${MODEL_PATH}"
  --tree_method eagle3
  --tree_fusion none
  --tree_model_path "${TREE_MODEL_PATH}"
  --answer-file evaluation/data/humaneval/rejection_boundary_phase1/answers.jsonl
  --max-new-tokens 512
  --max_cache_len "${MAX_CACHE_LEN}"
  --dtype "${DTYPE}"
)

if [ -n "${SAM_PATH}" ]; then
  cmd+=(--sam_path "${SAM_PATH}")
fi

"${cmd[@]}"
