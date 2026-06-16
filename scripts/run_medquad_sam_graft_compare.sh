#!/bin/bash
# Run a MedQuAD benchmark comparing EAGLE3 decoding with and without
# SAM sequence grafting.
#
# Outputs:
#   evaluation/data/medquad/model_answer/baseline.jsonl
#   evaluation/data/medquad/model_answer/eagle3_pure.jsonl
#   evaluation/data/medquad/model_answer/eagle3_no_graft.jsonl
#   evaluation/data/medquad/model_answer/eagle3_sam_graft.jsonl
#
# Configuration is read from .env when present:
#   MODEL_PATH, TREE_MODEL_PATH, CUDA_VISIBLE_DEVICES, SAM_PATH, DTYPE,
#   FEISHU_WEBHOOK_URL, NOTIFY_TAG, FORCE_RERUN

set -euo pipefail

cd $(dirname $0)/..

# Load .env before xtrace so webhook secrets do not appear in logs.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

MODEL_PATH=${MODEL_PATH:?MODEL_PATH must be set in .env or environment}
TREE_MODEL_PATH=${TREE_MODEL_PATH:?TREE_MODEL_PATH must be set in .env or environment}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
SAM_PATH=${SAM_PATH:-}
DTYPE=${DTYPE:-float16}
FEISHU_WEBHOOK_URL=${FEISHU_WEBHOOK_URL:-}
NOTIFY_TAG=${NOTIFY_TAG:-medquad-graft}
FORCE_RERUN=${FORCE_RERUN:-0}
export CUDA_VISIBLE_DEVICES

BENCH=medquad
NUM_QUESTIONS=${MEDQUAD_NUM_QUESTIONS:-200}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
SAMD_N_PREDICTS=${SAMD_N_PREDICTS:-40}
SAMD_LEN_THRESHOLD=${SAMD_LEN_THRESHOLD:-5}
SAMD_LEN_BIAS=${SAMD_LEN_BIAS:-5}
EAGLE3_TOTAL_TOKEN=${EAGLE3_TOTAL_TOKEN:-60}
EAGLE3_DEPTH=${EAGLE3_DEPTH:-7}
EAGLE3_TOP_K=${EAGLE3_TOP_K:-10}
ANSWER_DIR="evaluation/data/${BENCH}/model_answer"
BASE_JSONL="${ANSWER_DIR}/baseline.jsonl"
PURE_JSONL="${ANSWER_DIR}/eagle3_pure.jsonl"
NO_GRAFT_JSONL="${ANSWER_DIR}/eagle3_no_graft.jsonl"
GRAFT_JSONL="${ANSWER_DIR}/eagle3_sam_graft.jsonl"

notify() (
    set +x
    local msg="$1"
    local prefix="[${NOTIFY_TAG}]"
    echo "${prefix} ${msg}"
    if [ -z "${FEISHU_WEBHOOK_URL}" ]; then
        return 0
    fi
    local payload
    payload=$(python3 -c '
import json, sys
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
    printf '%dh%02dm%02ds' $((s/3600)) $((s%3600/60)) $((s%60))
}

jsonl_complete() {
    local path="$1"
    local expected_lines="$2"
    if [ ! -f "${path}" ]; then
        return 1
    fi
    local line_count
    line_count=$(wc -l < "${path}" | tr -d '[:space:]')
    [ "${line_count}" = "${expected_lines}" ]
}

run_step() {
    local label="$1"
    local output_path="$2"
    shift 2

    if [ "${FORCE_RERUN}" != "1" ] && jsonl_complete "${output_path}" "${NUM_QUESTIONS}"; then
        notify "skip ${label}: found complete ${output_path} (${NUM_QUESTIONS}/${NUM_QUESTIONS})"
        return 0
    fi

    if [ -f "${output_path}" ]; then
        local line_count
        line_count=$(wc -l < "${output_path}" | tr -d '[:space:]')
        notify "rerun ${label}: removing incomplete/stale ${output_path} (${line_count}/${NUM_QUESTIONS})"
        rm -f "${output_path}"
    fi

    local step_start
    step_start=$(date +%s)
    "$@"
    set +x
    notify "${label} done in $(fmt_elapsed $(($(date +%s) - step_start)))"
    set -x
}

trap 'rc=$?; notify "FAILED at line ${LINENO} (exit ${rc}). See log."; exit ${rc}' ERR

sam_args=()
if [ -n "${SAM_PATH}" ]; then
    sam_args+=(--sam_path "${SAM_PATH}")
fi

common_samd_args=(
    --template llama3
    --model-type llama3
    --bench-name "${BENCH}"
    --model-path "${MODEL_PATH}"
    --dtype "${DTYPE}"
    --tree_method eagle3
    --tree_model_path "${TREE_MODEL_PATH}"
    --eagle3_total_token "${EAGLE3_TOTAL_TOKEN}"
    --eagle3_depth "${EAGLE3_DEPTH}"
    --eagle3_top_k "${EAGLE3_TOP_K}"
    "${sam_args[@]}"
    --samd_n_predicts "${SAMD_N_PREDICTS}"
    --samd_len_bias "${SAMD_LEN_BIAS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max_cache_len "${MAX_CACHE_LEN}"
)

run_start=$(date +%s)
notify "START MedQuAD graft compare: questions=${NUM_QUESTIONS}, max_new=${MAX_NEW_TOKENS}"

set -x

python -m evaluation.medquad_prep --num_questions "${NUM_QUESTIONS}"
mkdir -p "${ANSWER_DIR}"

run_step "baseline" "${BASE_JSONL}" \
    python -m evaluation.inference_baseline \
    --template llama3 \
    --model-type llama3 \
    --bench-name "${BENCH}" \
    --model-path "${MODEL_PATH}" \
    --model-id baseline \
    --dtype "${DTYPE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"

run_step "pure_eagle3" "${PURE_JSONL}" \
    python -m evaluation.inference_samd \
    "${common_samd_args[@]}" \
    --model-id eagle3_pure \
    --tree_fusion none \
    --samd_len_threshold 999

run_step "no_graft" "${NO_GRAFT_JSONL}" \
    python -m evaluation.inference_samd \
    "${common_samd_args[@]}" \
    --model-id eagle3_no_graft \
    --tree_fusion none \
    --samd_len_threshold "${SAMD_LEN_THRESHOLD}"

run_step "sam_graft" "${GRAFT_JSONL}" \
    python -m evaluation.inference_samd \
    "${common_samd_args[@]}" \
    --model-id eagle3_sam_graft \
    --tree_fusion sam_sequence_graft \
    --samd_len_threshold "${SAMD_LEN_THRESHOLD}"

summary=$(python - "${BASE_JSONL}" "${PURE_JSONL}" "${NO_GRAFT_JSONL}" "${GRAFT_JSONL}" "${MODEL_PATH}" <<'PY'
import sys
import numpy as np
from evaluation.speed import speed

base_jsonl, pure_jsonl, no_graft_jsonl, graft_jsonl, tokenizer_path = sys.argv[1:6]
rows = []
for name, path in (
    ("pure_eagle3", pure_jsonl),
    ("no_graft", no_graft_jsonl),
    ("sam_graft", graft_jsonl),
):
    tps, base_tps, speedup, accepts = speed(
        path,
        base_jsonl,
        tokenizer_path,
        task="medquad",
        report=False,
    )
    rows.append((name, tps, base_tps, speedup, float(np.mean(accepts))))

print("| group | tok/s | baseline tok/s | speedup | mean accept |")
print("|---|---:|---:|---:|---:|")
for name, tps, base_tps, speedup, accept in rows:
    print(f"| {name} | {tps:.4f} | {base_tps:.4f} | {speedup:.4f} | {accept:.4f} |")

pure, no_graft, graft = rows
print("")
print(f"no_graft speedup delta vs pure_eagle3: {no_graft[3] - pure[3]:+.4f}")
print(f"no_graft mean-accept delta vs pure_eagle3: {no_graft[4] - pure[4]:+.4f}")
print(f"sam_graft speedup delta vs no_graft: {graft[3] - no_graft[3]:+.4f}")
print(f"sam_graft mean-accept delta vs no_graft: {graft[4] - no_graft[4]:+.4f}")
print(f"sam_graft speedup delta vs pure_eagle3: {graft[3] - pure[3]:+.4f}")
print(f"sam_graft mean-accept delta vs pure_eagle3: {graft[4] - pure[4]:+.4f}")
PY
)
echo "${summary}"

total_elapsed=$(fmt_elapsed $(($(date +%s) - run_start)))
notify "DONE in ${total_elapsed}

${summary}"
