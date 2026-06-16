#!/usr/bin/env bash
# Profile opt-in fusion overhead on MT-Bench (80 questions).
# Usage: bash scripts/profile_fusion_mt_bench.sh

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
BENCH_NAME=mt_bench
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
DTYPE=${DTYPE:-float16}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUTPUT_DIR=${PROFILE_OUTPUT_DIR:-evaluation/data/${BENCH_NAME}/profile_fusion_overhead}
MODEL_ID=${MODEL_ID:-naive_fusion_mt_bench_q0_80}
NOTIFY_TAG=${NOTIFY_TAG:-profile-fusion-mt-bench}

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
ANSWER_FILE="${OUTPUT_DIR}/${MODEL_ID}.jsonl"
PROFILE_JSON="${OUTPUT_DIR}/${MODEL_ID}.fusion_profile.json"
PROFILE_SUMMARY="${OUTPUT_DIR}/${MODEL_ID}.fusion_profile.txt"
ORACLE_RESULTS="${OUTPUT_DIR}/${MODEL_ID}.oracle_results.json"

run_start=$(date +%s)
trap 'rc=$?; notify "ERROR MT-Bench fusion profiling failed at line ${LINENO} (exit ${rc})"; exit ${rc}' ERR

notify "START MT-Bench fusion profiling: 80 questions, max_new_tokens=${MAX_NEW_TOKENS}, max_cache_len=${MAX_CACHE_LEN}"

cmd=(
    "${PYTHON_BIN}" -m evaluation.inference_samd
    --template llama3
    --model-type llama3
    --model-path "${MODEL_PATH}"
    --model-id "${MODEL_ID}"
    --bench-name "${BENCH_NAME}"
    --question-begin 0
    --question-end 80
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

# Print fusion overhead breakdown
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

# Run oracle analysis
echo ""
echo "Running oracle analysis..."
"${PYTHON_BIN}" evaluation/oracle_fusion_analysis.py \
    --trace-file "${PROFILE_JSON}" \
    --output-json "${ORACLE_RESULTS}"

# Print oracle results
"${PYTHON_BIN}" - "${ORACLE_RESULTS}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fin:
    data = json.load(fin)

print("")
print("Oracle analysis results")
print("oracle_json:", path)
print("")
print("{:<20} {:>10} {:>10}".format("method", "MAT", "gap"))
for result in data["results"]:
    method = result["method"]
    mat = result["mat"]
    gap = result.get("oracle_gap")
    if gap is not None:
        print("{:<20} {:>10.4f} {:>9.2f}%".format(method, mat, gap * 100))
    else:
        print("{:<20} {:>10.4f} {:>10}".format(method, mat, "+0.00%"))
PY

elapsed=$(fmt_elapsed $(($(date +%s) - run_start)))
notify "DONE MT-Bench fusion profiling in ${elapsed}. Oracle results: ${ORACLE_RESULTS}"

echo ""
echo "All results saved to:"
echo "  Answer file:     ${ANSWER_FILE}"
echo "  Profile JSON:    ${PROFILE_JSON}"
echo "  Profile summary: ${PROFILE_SUMMARY}"
echo "  Oracle results:  ${ORACLE_RESULTS}"
