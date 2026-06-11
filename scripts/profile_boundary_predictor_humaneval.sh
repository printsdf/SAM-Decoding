#!/usr/bin/env bash
# Capture enriched HumanEval fusion traces for boundary-predictor calibration.
# Run this inside the model environment checkout; this script does not ssh.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_PATH=${MODEL_PATH:-/root/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B}
SAM_PATH=${SAM_PATH:-}
BENCH_NAME=humaneval
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-164}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
DTYPE=${DTYPE:-float16}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-evaluation/data/${BENCH_NAME}/boundary_predictor_calibration}
MODEL_ID=${MODEL_ID:-boundary_calibration_humaneval_q${QUESTION_BEGIN}_${QUESTION_END}}
RUN_BOUNDARY_ANALYZER=${RUN_BOUNDARY_ANALYZER:-1}
BOUNDARY_MAX_THRESHOLDS=${BOUNDARY_MAX_THRESHOLDS:-256}
BOUNDARY_NODE_BUDGET=${BOUNDARY_NODE_BUDGET:-60}

if [ -z "${TRAIN_BEGIN:-}" ]; then
    TRAIN_BEGIN=${QUESTION_BEGIN}
fi
if [ -z "${TRAIN_END:-}" ]; then
    if [ "${QUESTION_END}" -lt 82 ]; then
        TRAIN_END=$(((QUESTION_BEGIN + QUESTION_END) / 2))
    else
        TRAIN_END=82
    fi
fi
if [ -z "${VALID_BEGIN:-}" ]; then
    VALID_BEGIN=${TRAIN_END}
fi
if [ -z "${VALID_END:-}" ]; then
    VALID_END=${QUESTION_END}
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if [ "${PYTHON_BIN}" = "python" ] && command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "ERROR: python executable not found: ${PYTHON_BIN}" >&2
        exit 127
    fi
fi

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_DIR}"
ANSWER_FILE="${OUTPUT_DIR}/${MODEL_ID}.jsonl"
PROFILE_JSON="${OUTPUT_DIR}/${MODEL_ID}.fusion_profile.json"
PROFILE_SUMMARY="${OUTPUT_DIR}/${MODEL_ID}.fusion_profile.txt"
ORACLE_RESULTS="${OUTPUT_DIR}/${MODEL_ID}.oracle_results.json"
REJECTION_BOUNDARY_RESULTS="${OUTPUT_DIR}/${MODEL_ID}.oracle_rejection_boundary.json"

echo "START boundary predictor HumanEval profile"
echo "bench_name: ${BENCH_NAME}"
echo "question_range: ${QUESTION_BEGIN}-${QUESTION_END}"
echo "model_id: ${MODEL_ID}"
echo "answer_file: ${ANSWER_FILE}"
echo "profile_json: ${PROFILE_JSON}"
echo "profile_summary: ${PROFILE_SUMMARY}"
echo "max_cache_len: ${MAX_CACHE_LEN}"

cmd=(
    "${PYTHON_BIN}" -m evaluation.inference_samd
    --template llama3
    --model-type llama3
    --model-path "${MODEL_PATH}"
    --model-id "${MODEL_ID}"
    --bench-name "${BENCH_NAME}"
    --question-begin "${QUESTION_BEGIN}"
    --question-end "${QUESTION_END}"
    --answer-file "${ANSWER_FILE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --dtype "${DTYPE}"
    --samd_n_predicts 40
    --samd_len_threshold 5
    --samd_len_bias 5
    --tree_method eagle3
    --tree_fusion none
    --fusion_mode naive
    --fusion_max_draft_tokens 60
    --tree_model_path "${TREE_MODEL_PATH}"
    --eagle3_total_token "${EAGLE3_TOTAL_TOKEN:-60}"
    --eagle3_depth "${EAGLE3_DEPTH:-7}"
    --eagle3_top_k "${EAGLE3_TOP_K:-10}"
    --max_cache_len "${MAX_CACHE_LEN}"
    --profile-fusion
    --fusion-profile-file "${PROFILE_JSON}"
    --fusion-profile-summary-file "${PROFILE_SUMMARY}"
)

if [ -n "${SAM_PATH}" ]; then
    cmd+=(--sam_path "${SAM_PATH}")
fi
if [ -n "${EAGLE3_TAIL_PATH:-}" ]; then
    cmd+=(--eagle3_tail_path "${EAGLE3_TAIL_PATH}")
fi
if [ -n "${EAGLE3_TAIL_TYPE:-}" ]; then
    cmd+=(--eagle3_tail_type "${EAGLE3_TAIL_TYPE}")
fi

"${cmd[@]}" "$@"

echo "Running oracle analysis"
"${PYTHON_BIN}" evaluation/oracle_fusion_analysis.py \
    --trace-file "${PROFILE_JSON}" \
    --output-json "${ORACLE_RESULTS}"

echo "Running rejection-boundary oracle analysis"
"${PYTHON_BIN}" evaluation/oracle_rejection_boundary.py \
    --trace-file "${PROFILE_JSON}" \
    --output "${REJECTION_BOUNDARY_RESULTS}"

if [ "${RUN_BOUNDARY_ANALYZER}" != "0" ]; then
    echo "Running boundary predictor calibration analyzer"
    "${PYTHON_BIN}" evaluation/analyze_boundary_predictor.py \
        --trace-file "${PROFILE_JSON}" \
        --answer-file "${ANSWER_FILE}" \
        --train-begin "${TRAIN_BEGIN}" \
        --train-end "${TRAIN_END}" \
        --valid-begin "${VALID_BEGIN}" \
        --valid-end "${VALID_END}" \
        --max-thresholds "${BOUNDARY_MAX_THRESHOLDS}" \
        --node-budget "${BOUNDARY_NODE_BUDGET}" \
        --output-dir "${OUTPUT_DIR}"
fi

echo "DONE boundary predictor HumanEval profile"
echo "answer_file: ${ANSWER_FILE}"
echo "profile_json: ${PROFILE_JSON}"
echo "profile_summary: ${PROFILE_SUMMARY}"
echo "oracle_results: ${ORACLE_RESULTS}"
echo "rejection_boundary_results: ${REJECTION_BOUNDARY_RESULTS}"
echo "output_dir: ${OUTPUT_DIR}"
