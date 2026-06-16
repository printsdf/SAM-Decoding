#!/usr/bin/env bash
# Evaluate rejection-boundary on full HumanEval (164 samples) with profiling

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-/root/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B}
SAM_PATH=${SAM_PATH:-}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
DTYPE=${DTYPE:-float16}

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p evaluation/data/humaneval/rejection_boundary_full

"${PYTHON_BIN}" -m evaluation.inference_samd \
  --template llama3 \
  --model-type llama3 \
  --model-id rejection_boundary_full \
  --bench-name humaneval \
  --question-begin 0 \
  --question-end 164 \
  --fusion_mode rejection_boundary \
  --rejection_conf_threshold 2.0 \
  --model-path "${MODEL_PATH}" \
  --tree_method eagle3 \
  --tree_fusion none \
  --tree_model_path "${TREE_MODEL_PATH}" \
  ${SAM_PATH:+--sam_path "${SAM_PATH}"} \
  --answer-file evaluation/data/humaneval/rejection_boundary_full/answers.jsonl \
  --max-new-tokens 512 \
  --max_cache_len "${MAX_CACHE_LEN}" \
  --dtype "${DTYPE}" \
  --profile-fusion \
  --fusion-profile-file evaluation/data/humaneval/rejection_boundary_full/profile.json \
  --fusion-profile-summary-file evaluation/data/humaneval/rejection_boundary_full/profile_summary.txt

echo ""
echo "Rejection-boundary complete. Results saved to:"
echo "  Answers: evaluation/data/humaneval/rejection_boundary_full/answers.jsonl"
echo "  Profile: evaluation/data/humaneval/rejection_boundary_full/profile.json"
echo "  Summary: evaluation/data/humaneval/rejection_boundary_full/profile_summary.txt"
