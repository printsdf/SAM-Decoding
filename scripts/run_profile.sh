#!/usr/bin/env bash
# Run the naive fusion profiler with cloudspace-friendly defaults.

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
BENCH_NAME=${BENCH_NAME:-mt_bench}
PROFILE_QUESTION_BEGIN=${PROFILE_QUESTION_BEGIN:-0}
PROFILE_NUM_QUESTIONS=${PROFILE_NUM_QUESTIONS:-10}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
DTYPE=${DTYPE:-float16}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PROFILE_OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-evaluation/data/${BENCH_NAME}/profile_naive_fusion}
NOTIFY_TAG=${NOTIFY_TAG:-profile-naive-fusion}

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

notify() (
    set +x
    local msg="$1"
    local prefix="[${NOTIFY_TAG}]"
    echo "${prefix} ${msg}"
    if [ -z "${FEISHU_WEBHOOK_URL:-}" ]; then
        return 0
    fi
    local payload
    payload=$("${PYTHON_BIN}" -c '
import json
import sys
prefix, body = sys.argv[1], sys.argv[2]
print(json.dumps({
    "msg_type": "text",
    "content": {"text": f"{prefix} {body}"},
}))
' "${prefix}" "${msg}") || { echo "[notify] payload build failed"; return 0; }
    curl -s -X POST -H "Content-Type: application/json" \
        -d "${payload}" \
        "${FEISHU_WEBHOOK_URL}" > /dev/null || echo "[notify] feishu post failed"
)

fmt_elapsed() {
    local s=$1
    printf '%dh%02dm%02ds' $((s / 3600)) $((s % 3600 / 60)) $((s % 60))
}

run_start=$(date +%s)
trap 'rc=$?; notify "ERROR profiling failed at line ${LINENO} (exit ${rc})"; exit ${rc}' ERR

notify "START naive fusion profiling: bench=${BENCH_NAME} q_begin=${PROFILE_QUESTION_BEGIN} num_questions=${PROFILE_NUM_QUESTIONS} max_new_tokens=${MAX_NEW_TOKENS} max_cache_len=${MAX_CACHE_LEN}"

cmd=(
    "${PYTHON_BIN}" scripts/profile_naive_fusion.py
    --model-path "${MODEL_PATH}"
    --tree-model-path "${TREE_MODEL_PATH}"
    --bench-name "${BENCH_NAME}"
    --question-begin "${PROFILE_QUESTION_BEGIN}"
    --num-questions "${PROFILE_NUM_QUESTIONS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max-cache-len "${MAX_CACHE_LEN}"
    --dtype "${DTYPE}"
    --output-dir "${PROFILE_OUTPUT_DIR}"
)

if [ -n "${SAM_PATH}" ]; then
    cmd+=(--sam-path "${SAM_PATH}")
fi

if [ -n "${EAGLE3_TAIL_PATH:-}" ]; then
    cmd+=(--eagle3-tail-path "${EAGLE3_TAIL_PATH}")
fi

if [ -n "${EAGLE3_TAIL_TYPE:-}" ]; then
    cmd+=(--eagle3-tail-type "${EAGLE3_TAIL_TYPE}")
fi

"${cmd[@]}" "$@"

elapsed=$(fmt_elapsed $(($(date +%s) - run_start)))
notify "DONE naive fusion profiling in ${elapsed}. Report dir: ${PROFILE_OUTPUT_DIR}"
