#!/bin/bash
# Run MedQA V_miss / accept_length comparison across three draft strategies:
#   - pure_eagle3  : SAM 切换关闭（len_threshold=999），全程走 EAGLE3 tree
#   - samd_eagle3  : SAM[EAGLE3] 混合，DynSAM + EAGLE3 tree 协同
#   - sam_only     : 纯 SAM (samd_sam_only/ 包，无 draft model)
#
# Outputs land at:
#   evaluation/data/medqa/model_answer/{pure_eagle3,samd_eagle3,sam_only}.jsonl
#
# Both EAGLE3 groups dump per-step diagnosis_traces (V_miss / verifier_target);
# the SAM-only group does not (no draft vocabulary → reachable is N/A).
#
# Configuration: copy .env.example to .env and edit paths / FEISHU_WEBHOOK_URL.

set -e

cd $(dirname $0)/..

# Load .env if present (gitignored; never committed).
# Done BEFORE `set -x` so secrets do not leak into logs.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Configuration with fallbacks for backward compatibility.
MODEL_PATH=${MODEL_PATH:-/root/Models/Meta-Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/Models/EAGLE3-LLaMA3.1-Instruct-8B}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FEISHU_WEBHOOK_URL=${FEISHU_WEBHOOK_URL:-}
NOTIFY_TAG=${NOTIFY_TAG:-medqa-vmiss}
export CUDA_VISIBLE_DEVICES

BENCH=medqa
NUM_QUESTIONS=80
MAX_NEW_TOKENS=512
# Cap SamdStaticCache size; Llama-3.1's 131072 default would alloc ~16 GiB
# of KV cache per generate call and OOM on 24 GiB GPUs.
MAX_CACHE_LEN=4096

# Send a Feishu text message; degrade to stdout-only when webhook unset.
# Defined as a subshell function so the inner `set +x` does not leak the
# webhook URL into the surrounding script's xtrace output.
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

# Format an elapsed seconds count as Hh MMm SSs.
fmt_elapsed() {
    local s=$1
    printf '%dh%02dm%02ds' $((s/3600)) $((s%3600/60)) $((s%60))
}

# On any error before set -e exits, send a Feishu alert. $LINENO is expanded
# at trap-trigger time because of the single-quoted body.
trap 'rc=$?; notify "❌ FAILED at line ${LINENO} (exit ${rc}). See log."; exit ${rc}' ERR

run_start=$(date +%s)
notify "▶ START num_questions=${NUM_QUESTIONS} max_new_tokens=${MAX_NEW_TOKENS} max_cache_len=${MAX_CACHE_LEN}"

set -x

# 1) Materialize the MedQA dataset (idempotent: rewrites question.jsonl each run).
python -m evaluation.medqa_prep --num_questions ${NUM_QUESTIONS}

# 2a) Pure EAGLE3
g_start=$(date +%s)
python -m evaluation.inference_samd \
    --template llama3 \
    --model-type llama3 \
    --bench-name ${BENCH} \
    --model-path ${MODEL_PATH} \
    --model-id pure_eagle3 \
    --tree_method eagle3 \
    --tree_model_path ${TREE_MODEL_PATH} \
    --samd_n_predicts 40 \
    --samd_len_threshold 999 \
    --samd_len_bias 5 \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max_cache_len ${MAX_CACHE_LEN} \
    --collect_diagnosis_trace
notify "✓ group 1/3 pure_eagle3 done in $(fmt_elapsed $(($(date +%s) - g_start)))"

# 2b) SAM[EAGLE3]: DynSAM 总是激活，不需要 static SAM pkl
g_start=$(date +%s)
python -m evaluation.inference_samd \
    --template llama3 \
    --model-type llama3 \
    --bench-name ${BENCH} \
    --model-path ${MODEL_PATH} \
    --model-id samd_eagle3 \
    --tree_method eagle3 \
    --tree_model_path ${TREE_MODEL_PATH} \
    --samd_n_predicts 40 \
    --samd_len_threshold 5 \
    --samd_len_bias 5 \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max_cache_len ${MAX_CACHE_LEN} \
    --collect_diagnosis_trace
notify "✓ group 2/3 samd_eagle3 done in $(fmt_elapsed $(($(date +%s) - g_start)))"

# 2c) SAM-only: DynSAM only, no trace flag
g_start=$(date +%s)
python -m evaluation.inference_sam_only \
    --template llama3 \
    --model-type llama3 \
    --bench-name ${BENCH} \
    --model-path ${MODEL_PATH} \
    --model-id sam_only \
    --samd_max_predicts 70 \
    --samd_len_bias 5 \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max_cache_len ${MAX_CACHE_LEN}
notify "✓ group 3/3 sam_only done in $(fmt_elapsed $(($(date +%s) - g_start)))"

# 3) Analyze and capture the markdown table for Feishu.
analyze_out=$(python -m evaluation.analyze_vmiss \
    --pure_eagle3 evaluation/data/${BENCH}/model_answer/pure_eagle3.jsonl \
    --samd_eagle3 evaluation/data/${BENCH}/model_answer/samd_eagle3.jsonl \
    --sam_only    evaluation/data/${BENCH}/model_answer/sam_only.jsonl)
echo "${analyze_out}"

total_elapsed=$(fmt_elapsed $(($(date +%s) - run_start)))
notify "✅ ALL DONE in ${total_elapsed}

${analyze_out}"
