#!/bin/bash
# Cross-domain speedup + V_miss benchmark for SAM-Decoding:
#   - Datasets:    mt_bench (FastChat 80q multi-turn) + medquad (NIH 200q free-form)
#   - Groups:      baseline / pure_eagle3 / samd_eagle3 / sam_only
#   - Phase split: phase1 trace OFF (wall-time speedup), phase2 trace ON (V_miss)
#
# Outputs land at evaluation/data/{mt_bench,medquad}/model_answer/{
#   baseline, pure_eagle3_p1, samd_eagle3_p1, sam_only_p1,
#   pure_eagle3_p2, samd_eagle3_p2 }.jsonl
#
# See .codestable/features/2026-05-25-bench-cross-domain-speedup/bench-cross-domain-speedup-design.md
# Configuration: copy .env.example to .env and edit paths / FEISHU_WEBHOOK_URL.

set -e

cd $(dirname $0)/..

# Load .env if present (gitignored; never committed).
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

MODEL_PATH=${MODEL_PATH:-/root/Models/Meta-Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/Models/EAGLE3-LLaMA3.1-Instruct-8B}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FEISHU_WEBHOOK_URL=${FEISHU_WEBHOOK_URL:-}
NOTIFY_TAG=${NOTIFY_TAG:-bench-cross-domain}
EAGLE_MT_BENCH_SRC=${EAGLE_MT_BENCH_SRC:-../EAGLE/eagle/data/mt_bench/question.jsonl}
export CUDA_VISIBLE_DEVICES

MEDQUAD_NUM_QUESTIONS=200
MAX_NEW_TOKENS=512
MAX_CACHE_LEN=4096

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

trap 'rc=$?; notify "❌ FAILED at line ${LINENO} (exit ${rc}). See log."; exit ${rc}' ERR

run_start=$(date +%s)
notify "▶ START cross-domain bench: 2 datasets × 4 groups × 2 phases, max_new_tokens=${MAX_NEW_TOKENS} max_cache_len=${MAX_CACHE_LEN}"

# ---------------------------------------------------------------------------
# Phase 0: fetch data (mt_bench cp + medquad_prep)
# ---------------------------------------------------------------------------
notify "▶ START fetch (mt_bench cp + medquad_prep ${MEDQUAD_NUM_QUESTIONS}q)"
phase_start=$(date +%s)
set -x

mkdir -p evaluation/data/mt_bench evaluation/data/medquad
if [ ! -f "${EAGLE_MT_BENCH_SRC}" ]; then
    set +x
    echo "ERROR: EAGLE mt_bench source not found at ${EAGLE_MT_BENCH_SRC}"
    echo "       set EAGLE_MT_BENCH_SRC in .env to override the path."
    exit 1
fi
cp "${EAGLE_MT_BENCH_SRC}" evaluation/data/mt_bench/question.jsonl

python -m evaluation.medquad_prep --num_questions ${MEDQUAD_NUM_QUESTIONS}

set +x
notify "✓ fetch done in $(fmt_elapsed $(($(date +%s) - phase_start)))"

# ---------------------------------------------------------------------------
# Phase 1: trace OFF, 4 groups × 2 benches = 8 inference runs
# ---------------------------------------------------------------------------
notify "▶ START phase1 (8 inference, trace OFF, baseline + 3 samd × 2 bench)"
phase_start=$(date +%s)

for BENCH in mt_bench medquad; do
    notify "  phase1 → bench=${BENCH}"

    # baseline
    g_start=$(date +%s)
    set -x
    python -m evaluation.inference_baseline \
        --template llama3 \
        --model-type llama3 \
        --bench-name ${BENCH} \
        --model-path ${MODEL_PATH} \
        --model-id baseline \
        --max-new-tokens ${MAX_NEW_TOKENS}
    set +x
    notify "    ✓ baseline ${BENCH} done in $(fmt_elapsed $(($(date +%s) - g_start)))"

    # pure_eagle3
    g_start=$(date +%s)
    set -x
    python -m evaluation.inference_samd \
        --template llama3 \
        --model-type llama3 \
        --bench-name ${BENCH} \
        --model-path ${MODEL_PATH} \
        --model-id pure_eagle3_p1 \
        --tree_method eagle3 \
        --tree_model_path ${TREE_MODEL_PATH} \
        --samd_n_predicts 40 \
        --samd_len_threshold 999 \
        --samd_len_bias 5 \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --max_cache_len ${MAX_CACHE_LEN}
    set +x
    notify "    ✓ pure_eagle3_p1 ${BENCH} done in $(fmt_elapsed $(($(date +%s) - g_start)))"

    # samd_eagle3
    g_start=$(date +%s)
    set -x
    python -m evaluation.inference_samd \
        --template llama3 \
        --model-type llama3 \
        --bench-name ${BENCH} \
        --model-path ${MODEL_PATH} \
        --model-id samd_eagle3_p1 \
        --tree_method eagle3 \
        --tree_model_path ${TREE_MODEL_PATH} \
        --samd_n_predicts 40 \
        --samd_len_threshold 5 \
        --samd_len_bias 5 \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --max_cache_len ${MAX_CACHE_LEN}
    set +x
    notify "    ✓ samd_eagle3_p1 ${BENCH} done in $(fmt_elapsed $(($(date +%s) - g_start)))"

    # sam_only
    g_start=$(date +%s)
    set -x
    python -m evaluation.inference_sam_only \
        --template llama3 \
        --model-type llama3 \
        --bench-name ${BENCH} \
        --model-path ${MODEL_PATH} \
        --model-id sam_only_p1 \
        --samd_max_predicts 70 \
        --samd_len_bias 5 \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --max_cache_len ${MAX_CACHE_LEN}
    set +x
    notify "    ✓ sam_only_p1 ${BENCH} done in $(fmt_elapsed $(($(date +%s) - g_start)))"
done
notify "✓ phase1 done in $(fmt_elapsed $(($(date +%s) - phase_start)))"

# ---------------------------------------------------------------------------
# Phase 1 analyze: speed.py × 6 (3 samd groups × 2 benches)
# ---------------------------------------------------------------------------
notify "▶ phase1 analyze (speed.py × 6: 3 groups × 2 bench)"
phase_start=$(date +%s)
speed_summary=""
for BENCH in mt_bench medquad; do
    BASE_JSONL=evaluation/data/${BENCH}/model_answer/baseline.jsonl
    for GROUP in pure_eagle3_p1 samd_eagle3_p1 sam_only_p1; do
        SAMD_JSONL=evaluation/data/${BENCH}/model_answer/${GROUP}.jsonl
        section_header="===== speed: bench=${BENCH} group=${GROUP} ====="
        echo "${section_header}"
        speed_out=$(python -m evaluation.speed \
            --file-path ${SAMD_JSONL} \
            --base-path ${BASE_JSONL} \
            --tokenizer-path ${MODEL_PATH})
        echo "${speed_out}"
        speed_summary="${speed_summary}
${section_header}
${speed_out}
"
    done
done
notify "✓ phase1 analyze done in $(fmt_elapsed $(($(date +%s) - phase_start)))"

# ---------------------------------------------------------------------------
# Phase 2: trace ON, EAGLE3 two groups × 2 benches = 4 inference runs
# ---------------------------------------------------------------------------
notify "▶ START phase2 (4 inference, trace ON, EAGLE3 only)"
phase_start=$(date +%s)

for BENCH in mt_bench medquad; do
    notify "  phase2 → bench=${BENCH}"

    g_start=$(date +%s)
    set -x
    python -m evaluation.inference_samd \
        --template llama3 \
        --model-type llama3 \
        --bench-name ${BENCH} \
        --model-path ${MODEL_PATH} \
        --model-id pure_eagle3_p2 \
        --tree_method eagle3 \
        --tree_model_path ${TREE_MODEL_PATH} \
        --samd_n_predicts 40 \
        --samd_len_threshold 999 \
        --samd_len_bias 5 \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --max_cache_len ${MAX_CACHE_LEN} \
        --collect_diagnosis_trace
    set +x
    notify "    ✓ pure_eagle3_p2 ${BENCH} done in $(fmt_elapsed $(($(date +%s) - g_start)))"

    g_start=$(date +%s)
    set -x
    python -m evaluation.inference_samd \
        --template llama3 \
        --model-type llama3 \
        --bench-name ${BENCH} \
        --model-path ${MODEL_PATH} \
        --model-id samd_eagle3_p2 \
        --tree_method eagle3 \
        --tree_model_path ${TREE_MODEL_PATH} \
        --samd_n_predicts 40 \
        --samd_len_threshold 5 \
        --samd_len_bias 5 \
        --max-new-tokens ${MAX_NEW_TOKENS} \
        --max_cache_len ${MAX_CACHE_LEN} \
        --collect_diagnosis_trace
    set +x
    notify "    ✓ samd_eagle3_p2 ${BENCH} done in $(fmt_elapsed $(($(date +%s) - g_start)))"
done
notify "✓ phase2 done in $(fmt_elapsed $(($(date +%s) - phase_start)))"

# ---------------------------------------------------------------------------
# Phase 2 analyze: analyze_vmiss × 2 benches (sam_only reuses phase1 file)
# ---------------------------------------------------------------------------
notify "▶ phase2 analyze (analyze_vmiss × 2 benches)"
phase_start=$(date +%s)
vmiss_summary=""
for BENCH in mt_bench medquad; do
    section_header="===== V_miss: bench=${BENCH} ====="
    echo "${section_header}"
    vmiss_out=$(python -m evaluation.analyze_vmiss \
        --pure_eagle3 evaluation/data/${BENCH}/model_answer/pure_eagle3_p2.jsonl \
        --samd_eagle3 evaluation/data/${BENCH}/model_answer/samd_eagle3_p2.jsonl \
        --sam_only    evaluation/data/${BENCH}/model_answer/sam_only_p1.jsonl)
    echo "${vmiss_out}"
    vmiss_summary="${vmiss_summary}
${section_header}
${vmiss_out}
"
done
notify "✓ phase2 analyze done in $(fmt_elapsed $(($(date +%s) - phase_start)))"

total_elapsed=$(fmt_elapsed $(($(date +%s) - run_start)))
notify "✅ ALL DONE in ${total_elapsed}
${vmiss_summary}"
