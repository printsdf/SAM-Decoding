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
BOUNDARY_PREDICTOR_SUITE=${BOUNDARY_PREDICTOR_SUITE:-default}
BOUNDARY_THETA_GRID=${BOUNDARY_THETA_GRID:-}
BOUNDARY_REACHABLE_QUANTILES=${BOUNDARY_REACHABLE_QUANTILES:-}
BOUNDARY_REQUIRE_RAW_DRAFT_LOGITS=${BOUNDARY_REQUIRE_RAW_DRAFT_LOGITS:-0}
SAMD_N_PREDICTS=${SAMD_N_PREDICTS:-40}
SAMD_LEN_THRESHOLD=${SAMD_LEN_THRESHOLD:-5}
SAMD_LEN_BIAS=${SAMD_LEN_BIAS:-5}
FUSION_MAX_DRAFT_TOKENS=${FUSION_MAX_DRAFT_TOKENS:-60}
EAGLE3_TOTAL_TOKEN=${EAGLE3_TOTAL_TOKEN:-60}
EAGLE3_DEPTH=${EAGLE3_DEPTH:-7}
EAGLE3_TOP_K=${EAGLE3_TOP_K:-10}

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
PROVENANCE_FILE="${OUTPUT_DIR}/${MODEL_ID}.provenance.txt"

missing_required=0
require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required local file missing: $1" >&2
        missing_required=1
    fi
}

require_file evaluation/oracle_fusion_analysis.py
require_file evaluation/oracle_rejection_boundary.py
if [ "${RUN_BOUNDARY_ANALYZER}" != "0" ]; then
    require_file evaluation/analyze_boundary_predictor.py
fi
if [ ! -f evaluation/oracle_depth_decoupled.py ]; then
    echo "WARNING: optional depth-decoupled oracle missing; oracle_fusion_analysis.py will skip that payload if requested." >&2
fi
if [ "${missing_required}" != "0" ]; then
    echo "ERROR: preflight failed before model execution; sync or track the missing analysis files first." >&2
    exit 2
fi

hash_file() {
    if [ ! -f "$1" ]; then
        echo "missing"
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1"
    else
        echo "sha256_unavailable $1"
    fi
}

write_provenance() {
    local question_file="evaluation/data/${BENCH_NAME}/question.jsonl"
    {
        echo "script: scripts/profile_boundary_predictor_humaneval.sh"
        echo "cwd: $(pwd)"
        echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
        if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            echo "git_status_short_count: $(git status --short | wc -l | tr -d ' ')"
            echo "git_status_short:"
            git status --short | sed -n '1,40p'
        else
            echo "git_status_short_count: unknown"
        fi
        echo "python_bin: ${PYTHON_BIN}"
        echo "bench_name: ${BENCH_NAME}"
        echo "question_range: ${QUESTION_BEGIN}-${QUESTION_END}"
        echo "train_range: ${TRAIN_BEGIN}-${TRAIN_END}"
        echo "valid_range: ${VALID_BEGIN}-${VALID_END}"
        echo "question_file: ${question_file}"
        if [ -f "${question_file}" ]; then
            echo "question_file_rows: $(wc -l < "${question_file}" | tr -d ' ')"
            echo "question_file_sha256: $(hash_file "${question_file}")"
        else
            echo "question_file_rows: missing"
            echo "question_file_sha256: missing"
        fi
        echo "model_path: ${MODEL_PATH}"
        echo "tree_model_path: ${TREE_MODEL_PATH}"
        echo "sam_path: ${SAM_PATH:-}"
        echo "max_new_tokens: ${MAX_NEW_TOKENS}"
        echo "max_cache_len: ${MAX_CACHE_LEN}"
        echo "dtype: ${DTYPE}"
        echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
        echo "samd_n_predicts: ${SAMD_N_PREDICTS}"
        echo "samd_len_threshold: ${SAMD_LEN_THRESHOLD}"
        echo "samd_len_bias: ${SAMD_LEN_BIAS}"
        echo "fusion_max_draft_tokens: ${FUSION_MAX_DRAFT_TOKENS}"
        echo "eagle3_total_token: ${EAGLE3_TOTAL_TOKEN}"
        echo "eagle3_depth: ${EAGLE3_DEPTH}"
        echo "eagle3_top_k: ${EAGLE3_TOP_K}"
        echo "boundary_analyzer_enabled: ${RUN_BOUNDARY_ANALYZER}"
        echo "boundary_predictor_suite: ${BOUNDARY_PREDICTOR_SUITE}"
        echo "boundary_max_thresholds: ${BOUNDARY_MAX_THRESHOLDS}"
        echo "boundary_node_budget: ${BOUNDARY_NODE_BUDGET}"
        echo "boundary_theta_grid: ${BOUNDARY_THETA_GRID}"
        echo "boundary_reachable_quantiles: ${BOUNDARY_REACHABLE_QUANTILES}"
        echo "boundary_require_raw_draft_logits: ${BOUNDARY_REQUIRE_RAW_DRAFT_LOGITS}"
    } > "${PROVENANCE_FILE}"
}

write_provenance

echo "START boundary predictor HumanEval profile"
echo "bench_name: ${BENCH_NAME}"
echo "question_range: ${QUESTION_BEGIN}-${QUESTION_END}"
echo "model_id: ${MODEL_ID}"
echo "answer_file: ${ANSWER_FILE}"
echo "profile_json: ${PROFILE_JSON}"
echo "profile_summary: ${PROFILE_SUMMARY}"
echo "provenance_file: ${PROVENANCE_FILE}"
echo "max_cache_len: ${MAX_CACHE_LEN}"
echo "boundary_predictor_suite: ${BOUNDARY_PREDICTOR_SUITE}"

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
    --samd_n_predicts "${SAMD_N_PREDICTS}"
    --samd_len_threshold "${SAMD_LEN_THRESHOLD}"
    --samd_len_bias "${SAMD_LEN_BIAS}"
    --tree_method eagle3
    --tree_fusion none
    --fusion_mode naive
    --fusion_max_draft_tokens "${FUSION_MAX_DRAFT_TOKENS}"
    --tree_model_path "${TREE_MODEL_PATH}"
    --eagle3_total_token "${EAGLE3_TOTAL_TOKEN}"
    --eagle3_depth "${EAGLE3_DEPTH}"
    --eagle3_top_k "${EAGLE3_TOP_K}"
    --max_cache_len "${MAX_CACHE_LEN}"
    --profile-fusion
    --fusion-profile-file "${PROFILE_JSON}"
    --fusion-profile-summary-file "${PROFILE_SUMMARY}"
)

if [ -n "${SAM_PATH}" ]; then
    cmd+=(--sam_path "${SAM_PATH}")
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
    analyzer_help=$("${PYTHON_BIN}" evaluation/analyze_boundary_predictor.py --help 2>&1 || true)
    analyzer_cmd=(
        "${PYTHON_BIN}" evaluation/analyze_boundary_predictor.py
        --trace-file "${PROFILE_JSON}"
        --answer-file "${ANSWER_FILE}"
        --train-begin "${TRAIN_BEGIN}"
        --train-end "${TRAIN_END}"
        --valid-begin "${VALID_BEGIN}"
        --valid-end "${VALID_END}"
        --max-thresholds "${BOUNDARY_MAX_THRESHOLDS}"
        --node-budget "${BOUNDARY_NODE_BUDGET}"
        --output-dir "${OUTPUT_DIR}"
    )
    if printf '%s\n' "${analyzer_help}" | grep -q -- "--predictor-suite"; then
        analyzer_cmd+=(--predictor-suite "${BOUNDARY_PREDICTOR_SUITE}")
    elif [ "${BOUNDARY_PREDICTOR_SUITE}" != "default" ]; then
        echo "WARNING: analyzer does not support --predictor-suite; requested suite ${BOUNDARY_PREDICTOR_SUITE} will not run." >&2
    fi
    if [ -n "${BOUNDARY_THETA_GRID}" ]; then
        if printf '%s\n' "${analyzer_help}" | grep -q -- "--theta-grid"; then
            analyzer_cmd+=(--theta-grid "${BOUNDARY_THETA_GRID}")
        else
            echo "WARNING: analyzer does not support --theta-grid; ignoring BOUNDARY_THETA_GRID." >&2
        fi
    fi
    if [ -n "${BOUNDARY_REACHABLE_QUANTILES}" ]; then
        if printf '%s\n' "${analyzer_help}" | grep -q -- "--reachable-quantiles"; then
            analyzer_cmd+=(--reachable-quantiles "${BOUNDARY_REACHABLE_QUANTILES}")
        else
            echo "WARNING: analyzer does not support --reachable-quantiles; ignoring BOUNDARY_REACHABLE_QUANTILES." >&2
        fi
    fi
    if [ "${BOUNDARY_REQUIRE_RAW_DRAFT_LOGITS}" != "0" ]; then
        if printf '%s\n' "${analyzer_help}" | grep -q -- "--require-raw-draft-logits"; then
            analyzer_cmd+=(--require-raw-draft-logits)
        else
            echo "WARNING: analyzer does not support --require-raw-draft-logits; raw-logit schema will not be enforced." >&2
        fi
    fi
    "${analyzer_cmd[@]}"
fi

echo "DONE boundary predictor HumanEval profile"
echo "answer_file: ${ANSWER_FILE}"
echo "profile_json: ${PROFILE_JSON}"
echo "profile_summary: ${PROFILE_SUMMARY}"
echo "oracle_results: ${ORACLE_RESULTS}"
echo "rejection_boundary_results: ${REJECTION_BOUNDARY_RESULTS}"
echo "provenance_file: ${PROVENANCE_FILE}"
echo "output_dir: ${OUTPUT_DIR}"
