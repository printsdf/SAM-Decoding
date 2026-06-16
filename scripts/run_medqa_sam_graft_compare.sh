#!/bin/bash
# Run a MedQA benchmark comparing EAGLE3 decoding with and without SAM graft.
#
# Outputs:
#   evaluation/data/medqa/model_answer/baseline.jsonl
#   evaluation/data/medqa/model_answer/eagle3_pure.jsonl
#   evaluation/data/medqa/model_answer/eagle3_no_graft.jsonl
#   evaluation/data/medqa/model_answer/eagle3_sam_graft.jsonl
#
# Configuration is read from .env when present:
#   MODEL_PATH, TREE_MODEL_PATH, CUDA_VISIBLE_DEVICES, SAM_PATH, DTYPE,
#   FEISHU_WEBHOOK_URL, NOTIFY_TAG

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
NOTIFY_TAG=${NOTIFY_TAG:-medqa-graft}
export CUDA_VISIBLE_DEVICES

BENCH=medqa
NUM_QUESTIONS=${MEDQA_NUM_QUESTIONS:-80}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
SAMD_N_PREDICTS=${SAMD_N_PREDICTS:-40}
SAMD_LEN_THRESHOLD=${SAMD_LEN_THRESHOLD:-5}
SAMD_LEN_BIAS=${SAMD_LEN_BIAS:-5}
EAGLE3_TOTAL_TOKEN=${EAGLE3_TOTAL_TOKEN:-60}
EAGLE3_DEPTH=${EAGLE3_DEPTH:-7}
EAGLE3_TOP_K=${EAGLE3_TOP_K:-10}

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

trap 'rc=$?; notify "FAILED at line ${LINENO} (exit ${rc}). See log."; exit ${rc}' ERR

sam_args=()
if [ -n "${SAM_PATH}" ]; then
    sam_args+=(--sam_path "${SAM_PATH}")
fi

common_eagle3_args=(
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
    --samd_n_predicts "${SAMD_N_PREDICTS}"
    --samd_len_bias "${SAMD_LEN_BIAS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --max_cache_len "${MAX_CACHE_LEN}"
)

common_samd_args=(
    "${common_eagle3_args[@]}"
    "${sam_args[@]}"
)

run_start=$(date +%s)
notify "START MedQA compare: questions=${NUM_QUESTIONS}, max_new=${MAX_NEW_TOKENS}"

set -x

python -m evaluation.medqa_prep --num_questions "${NUM_QUESTIONS}"
rm -f \
    "evaluation/data/${BENCH}/model_answer/baseline.jsonl" \
    "evaluation/data/${BENCH}/model_answer/eagle3_pure.jsonl" \
    "evaluation/data/${BENCH}/model_answer/eagle3_no_graft.jsonl" \
    "evaluation/data/${BENCH}/model_answer/eagle3_sam_graft.jsonl"

g_start=$(date +%s)
python -m evaluation.inference_baseline \
    --template llama3 \
    --model-type llama3 \
    --bench-name "${BENCH}" \
    --model-path "${MODEL_PATH}" \
    --model-id baseline \
    --dtype "${DTYPE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"
set +x
notify "baseline done in $(fmt_elapsed $(($(date +%s) - g_start)))"
set -x

g_start=$(date +%s)
python -m evaluation.inference_samd \
    "${common_eagle3_args[@]}" \
    --model-id eagle3_pure \
    --tree_fusion none \
    --samd_len_threshold 999
set +x
notify "pure_eagle3 done in $(fmt_elapsed $(($(date +%s) - g_start)))"
set -x

g_start=$(date +%s)
python -m evaluation.inference_samd \
    "${common_samd_args[@]}" \
    --model-id eagle3_no_graft \
    --tree_fusion none \
    --samd_len_threshold "${SAMD_LEN_THRESHOLD}"
set +x
notify "no_graft done in $(fmt_elapsed $(($(date +%s) - g_start)))"
set -x

g_start=$(date +%s)
python -m evaluation.inference_samd \
    "${common_samd_args[@]}" \
    --model-id eagle3_sam_graft \
    --tree_fusion sam_sequence_graft \
    --samd_len_threshold "${SAMD_LEN_THRESHOLD}"
set +x
notify "sam_graft done in $(fmt_elapsed $(($(date +%s) - g_start)))"

BASE_JSONL="evaluation/data/${BENCH}/model_answer/baseline.jsonl"
PURE_JSONL="evaluation/data/${BENCH}/model_answer/eagle3_pure.jsonl"
NO_GRAFT_JSONL="evaluation/data/${BENCH}/model_answer/eagle3_no_graft.jsonl"
GRAFT_JSONL="evaluation/data/${BENCH}/model_answer/eagle3_sam_graft.jsonl"

summary=$(python - "${BASE_JSONL}" "${PURE_JSONL}" "${NO_GRAFT_JSONL}" "${GRAFT_JSONL}" "${MODEL_PATH}" <<'PY'
import sys
import numpy as np
from evaluation.speed import speed

base_jsonl, pure_jsonl, no_graft_jsonl, graft_jsonl, tokenizer_path = sys.argv[1:6]
rows = []
for name, path in (
    ("eagle3_pure", pure_jsonl),
    ("eagle3_no_graft", no_graft_jsonl),
    ("eagle3_sam_graft", graft_jsonl),
):
    tps, base_tps, speedup, accepts = speed(
        path,
        base_jsonl,
        tokenizer_path,
        task="medqa",
        report=False,
    )
    rows.append((name, float(np.mean(accepts)), tps, base_tps, speedup))

print("| group | mean accept | tok/s | baseline tok/s | speedup |")
print("|---|---:|---:|---:|---:|")
for name, accept, tps, base_tps, speedup in rows:
    print(f"| {name} | {accept:.4f} | {tps:.4f} | {base_tps:.4f} | {speedup:.4f} |")

pure, no_graft, graft = rows
print("")
print(f"eagle3_no_graft speedup delta vs eagle3_pure: {no_graft[4] - pure[4]:+.4f}")
print(f"eagle3_no_graft accept delta vs eagle3_pure: {no_graft[1] - pure[1]:+.4f}")
print(f"eagle3_sam_graft speedup delta vs eagle3_no_graft: {graft[4] - no_graft[4]:+.4f}")
print(f"eagle3_sam_graft accept delta vs eagle3_no_graft: {graft[1] - no_graft[1]:+.4f}")
print(f"eagle3_sam_graft speedup delta vs eagle3_pure: {graft[4] - pure[4]:+.4f}")
print(f"eagle3_sam_graft accept delta vs eagle3_pure: {graft[1] - pure[1]:+.4f}")
PY
)
echo "${summary}"

total_elapsed=$(fmt_elapsed $(($(date +%s) - run_start)))
notify "DONE in ${total_elapsed}

${summary}"
