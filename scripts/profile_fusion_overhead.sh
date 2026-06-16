#!/usr/bin/env bash
# Profile opt-in fusion overhead on a small MT-Bench slice.

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
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
DTYPE=${DTYPE:-float16}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-evaluation/data/${BENCH_NAME}/profile_fusion_overhead}
MODEL_ID=${MODEL_ID:-naive_fusion_overhead_q${PROFILE_QUESTION_BEGIN}_${PROFILE_NUM_QUESTIONS}}
NOTIFY_TAG=${NOTIFY_TAG:-profile-fusion-overhead}

if [ "$#" -ge 2 ] && [[ "$1" =~ ^[0-9]+$ ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
    PROFILE_QUESTION_BEGIN=$1
    profile_question_end=$2
    if [ "${profile_question_end}" -le "${PROFILE_QUESTION_BEGIN}" ]; then
        echo "ERROR: question_end must be greater than question_begin" >&2
        exit 2
    fi
    PROFILE_NUM_QUESTIONS=$((profile_question_end - PROFILE_QUESTION_BEGIN))
    MODEL_ID="naive_fusion_overhead_q${PROFILE_QUESTION_BEGIN}_${profile_question_end}"
    shift 2
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

mkdir -p "${OUTPUT_DIR}"
question_end=$((PROFILE_QUESTION_BEGIN + PROFILE_NUM_QUESTIONS))
ANSWER_FILE="${OUTPUT_DIR}/${MODEL_ID}.jsonl"
PROFILE_JSON="${OUTPUT_DIR}/${MODEL_ID}.fusion_profile.json"
PROFILE_SUMMARY="${OUTPUT_DIR}/${MODEL_ID}.fusion_profile.txt"

run_start=$(date +%s)
trap 'rc=$?; notify "ERROR fusion overhead profiling failed at line ${LINENO} (exit ${rc})"; exit ${rc}' ERR

notify "START fusion overhead profiling: bench=${BENCH_NAME} q_begin=${PROFILE_QUESTION_BEGIN} num_questions=${PROFILE_NUM_QUESTIONS} max_new_tokens=${MAX_NEW_TOKENS} max_cache_len=${MAX_CACHE_LEN}"

cmd=(
    "${PYTHON_BIN}" -m evaluation.inference_samd
    --template llama3
    --model-type llama3
    --model-path "${MODEL_PATH}"
    --model-id "${MODEL_ID}"
    --bench-name "${BENCH_NAME}"
    --question-begin "${PROFILE_QUESTION_BEGIN}"
    --question-end "${question_end}"
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

"${cmd[@]}" "$@"

"${PYTHON_BIN}" - "${PROFILE_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fin:
    data = json.load(fin)
summary = data["summary"]
timings = summary["timings"]
pcts = summary["timing_pct"]
avg = summary["avg_step_timings"]

print("")
print("Fusion overhead breakdown")
print("profile_json:", path)
print("steps:", summary["steps"])
print("fusion_overhead_pct: {:.2f}%".format(summary["fusion_overhead_pct"]))
print("memory_alloc_mib: {:.2f}".format(summary["memory_alloc"] / (1024.0 * 1024.0)))
print("")
print("{:<16} {:>12} {:>12} {:>12}".format("phase", "seconds", "pct_total", "avg_ms"))
for phase in ("draft_eagle", "draft_sam", "fusion_logic", "verify", "total"):
    print("{:<16} {:>12.6f} {:>11.2f}% {:>12.3f}".format(
        phase,
        timings.get(phase, 0.0),
        pcts.get(phase, 0.0),
        avg.get(phase, 0.0) * 1000.0,
    ))
PY

elapsed=$(fmt_elapsed $(($(date +%s) - run_start)))
notify "DONE fusion overhead profiling in ${elapsed}. Report: ${PROFILE_SUMMARY}"
