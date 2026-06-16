#!/usr/bin/env bash
# Profile opt-in fusion overhead on a small MT-Bench slice.
# This script is retained for the no-sidecar oracle gate.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

resolve_python() {
    local candidates=(
        "${NO_TAIL_PYTHON_BIN:-}"
        "${PYTHON_BIN:-}"
        "/opt/conda/bin/python3"
        "python"
        "python3"
    )
    local candidate resolved
    for candidate in "${candidates[@]}"; do
        if [ -z "${candidate}" ]; then
            continue
        fi
        if [ -x "${candidate}" ] && [ ! -d "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        if resolved=$(command -v "${candidate}" 2>/dev/null); then
            if [ -x "${resolved}" ] && [ ! -d "${resolved}" ]; then
                printf '%s\n' "${resolved}"
                return 0
            fi
        fi
    done
    echo "ERROR: no usable python executable found" >&2
    return 127
}

PYTHON_BIN=$(resolve_python)
MODEL_PATH=${MODEL_PATH:-${HOME}/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-${HOME}/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B}
SAM_PATH=${SAM_PATH:-}
BENCH_NAME=${BENCH_NAME:-mt_bench}
PROFILE_QUESTION_BEGIN=${PROFILE_QUESTION_BEGIN:-0}
PROFILE_NUM_QUESTIONS=${PROFILE_NUM_QUESTIONS:-10}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
DTYPE=${DTYPE:-float16}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-evaluation/data/${BENCH_NAME}/profile_fusion_overhead_no_tail}
RUN_TAG=${PROFILE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
MODEL_ID=${MODEL_ID:-}
NOTIFY_TAG=${NOTIFY_TAG:-profile-fusion-overhead-no-tail}

if [ "$#" -ge 2 ] && [[ "$1" =~ ^[0-9]+$ ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
    PROFILE_QUESTION_BEGIN=$1
    profile_question_end=$2
    if [ "${profile_question_end}" -le "${PROFILE_QUESTION_BEGIN}" ]; then
        echo "ERROR: question_end must be greater than question_begin" >&2
        exit 2
    fi
    PROFILE_NUM_QUESTIONS=$((profile_question_end - PROFILE_QUESTION_BEGIN))
    shift 2
fi

if [ -z "${MODEL_ID}" ]; then
    question_end_for_id=$((PROFILE_QUESTION_BEGIN + PROFILE_NUM_QUESTIONS))
    MODEL_ID=naive_fusion_overhead_no_tail_q${PROFILE_QUESTION_BEGIN}_${question_end_for_id}_${RUN_TAG}
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
' "${prefix}" "${msg}" < /dev/null) || { echo "[notify] payload build failed"; return 0; }
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
RUN_LOG="${OUTPUT_DIR}/${MODEL_ID}.run.log"
ORACLE_JSON="${OUTPUT_DIR}/${MODEL_ID}.oracle_results.json"
ORACLE_LOG="${OUTPUT_DIR}/${MODEL_ID}.oracle.log"

run_start=$(date +%s)
trap 'rc=$?; notify "ERROR no-tail fusion overhead profiling failed at line ${LINENO} (exit ${rc})"; exit ${rc}' ERR

{
    echo "START no-tail fusion overhead profiling $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "python_bin: ${PYTHON_BIN}"
    echo "bench: ${BENCH_NAME}"
    echo "question_begin: ${PROFILE_QUESTION_BEGIN}"
    echo "question_end: ${question_end}"
    echo "max_new_tokens: ${MAX_NEW_TOKENS}"
    echo "max_cache_len: ${MAX_CACHE_LEN}"
    echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES}"
    echo "model_path: ${MODEL_PATH}"
    echo "tree_model_path: ${TREE_MODEL_PATH}"
    echo "answer_file: ${ANSWER_FILE}"
    echo "profile_json: ${PROFILE_JSON}"
    echo "profile_summary: ${PROFILE_SUMMARY}"
    echo "oracle_json: ${ORACLE_JSON}"
} | tee "${RUN_LOG}"

notify "START no-tail fusion overhead profiling: bench=${BENCH_NAME} q_begin=${PROFILE_QUESTION_BEGIN} num_questions=${PROFILE_NUM_QUESTIONS} max_new_tokens=${MAX_NEW_TOKENS} max_cache_len=${MAX_CACHE_LEN}"

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
    --fusion_dedup_strategy max_score
    --fusion_truncate_strategy score
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

"${cmd[@]}" "$@" < /dev/null 2>&1 | tee -a "${RUN_LOG}"

"${PYTHON_BIN}" evaluation/oracle_fusion_analysis.py \
    --trace-file "${PROFILE_JSON}" \
    --output-json "${ORACLE_JSON}" \
    < /dev/null 2>&1 | tee "${ORACLE_LOG}"

"${PYTHON_BIN}" - "${PROFILE_JSON}" "${ORACLE_JSON}" <<'PY'
import json
import sys

profile_path, oracle_path = sys.argv[1], sys.argv[2]
with open(profile_path, "r", encoding="utf-8") as fin:
    profile = json.load(fin)
summary = profile["summary"]
timings = summary["timings"]
pcts = summary["timing_pct"]
avg = summary["avg_step_timings"]

with open(oracle_path, "r", encoding="utf-8") as fin:
    oracle = json.load(fin)

print("")
print("No-tail fusion overhead breakdown")
print("profile_json:", profile_path)
print("oracle_json:", oracle_path)
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

print("")
print("Oracle gaps")
for row in oracle.get("methods", []):
    gap = row.get("oracle_gap")
    gap_text = "n/a" if gap is None else "{:+.2%}".format(gap)
    print("{:<20} MAT={:.4f} nodes={:.2f} gap={}".format(
        row.get("method", ""),
        row.get("mat", 0.0),
        row.get("nodes", 0.0),
        gap_text,
    ))
PY

elapsed=$(fmt_elapsed $(($(date +%s) - run_start)))
notify "DONE no-tail fusion overhead profiling in ${elapsed}. Profile: ${PROFILE_SUMMARY}. Oracle: ${ORACLE_JSON}"
