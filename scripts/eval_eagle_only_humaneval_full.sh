#!/usr/bin/env bash
# Evaluate eagle-only baseline on full HumanEval (164 samples)

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-/root/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
DTYPE=${DTYPE:-float16}

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p evaluation/data/humaneval/eagle_only_baseline

"${PYTHON_BIN}" -m evaluation.inference_samd \
  --template llama3 \
  --model-type llama3 \
  --model-id eagle_only_baseline_full \
  --bench-name humaneval \
  --question-begin 0 \
  --question-end 164 \
  --fusion_mode none \
  --model-path "${MODEL_PATH}" \
  --tree_method eagle3 \
  --tree_fusion none \
  --tree_model_path "${TREE_MODEL_PATH}" \
  --answer-file evaluation/data/humaneval/eagle_only_baseline/answers_full.jsonl \
  --max-new-tokens 512 \
  --max_cache_len "${MAX_CACHE_LEN}" \
  --dtype "${DTYPE}"

echo ""
echo "Eagle-only baseline complete. Answer file saved to:"
echo "  evaluation/data/humaneval/eagle_only_baseline/answers_full.jsonl"
