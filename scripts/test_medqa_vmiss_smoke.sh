#!/bin/bash
# One-shot smoke test of the entire medqa-vmiss-eval feature on GPU.
# Coverage: every step's runtime exit signal in a few minutes.
#
#   step 1+2: tests/test_samd_diagnosis.py (unit + SamdModel trace on/off)
#   step 4  : evaluation/medqa_prep.py     (5-question smoke subset)
#   step 3  : evaluation/inference_samd.py x2 + inference_sam_only.py x1
#             on first 2 questions, with --collect_diagnosis_trace for eagle3 groups
#   step 6  : evaluation/analyze_vmiss.py  (markdown table + sanity counts)
#
# Smoke uses bench-name=medqa-smoke so it does NOT overwrite the full 80-question
# data set at evaluation/data/medqa/.
#
# After this PASSes, run the full benchmark with:
#   bash scripts/run_medqa_vmiss_eval.sh   # ~1.5 hours, 80 questions, 3 groups.

set -e
set -x

cd $(dirname $0)/..

devices=0
MODEL_PATH=/root/Models/Meta-Llama-3.1-8B-Instruct
TREE_MODEL_PATH=/root/Models/EAGLE3-LLaMA3.1-Instruct-8B
BENCH=medqa-smoke
SMOKE_N=2
MAX_NEW_TOKENS=256
# Cap SamdStaticCache size; Llama-3.1's 131072 default would alloc ~16 GiB
# of KV cache per generate call and OOM on 24 GiB GPUs.
MAX_CACHE_LEN=4096

# ---- Step 1+2: unit + integration trace on/off ----
CUDA_VISIBLE_DEVICES=${devices} \
    python -m tests.test_samd_diagnosis \
    --mode both \
    --model_path ${MODEL_PATH} \
    --tree_model_path ${TREE_MODEL_PATH}

# ---- Step 4: materialize 5 MedQA questions into the smoke bench dir ----
python -m evaluation.medqa_prep \
    --num_questions 5 \
    --out_path evaluation/data/${BENCH}/question.jsonl

# ---- Step 3 smoke: pure EAGLE3 (no SAM) on SMOKE_N questions, with trace ----
CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_samd \
    --template llama3 \
    --model-type llama3 \
    --bench-name ${BENCH} \
    --model-path ${MODEL_PATH} \
    --model-id smoke_pure_eagle3 \
    --tree_method eagle3 \
    --tree_model_path ${TREE_MODEL_PATH} \
    --samd_n_predicts 40 \
    --samd_len_threshold 999 \
    --samd_len_bias 5 \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max_cache_len ${MAX_CACHE_LEN} \
    --question-end ${SMOKE_N} \
    --collect_diagnosis_trace

# ---- Step 3 smoke: SAM[EAGLE3] hybrid on SMOKE_N questions, with trace ----
# DynSAM is always active; no static SAM pkl is needed for the hybrid story.
CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_samd \
    --template llama3 \
    --model-type llama3 \
    --bench-name ${BENCH} \
    --model-path ${MODEL_PATH} \
    --model-id smoke_samd_eagle3 \
    --tree_method eagle3 \
    --tree_model_path ${TREE_MODEL_PATH} \
    --samd_n_predicts 40 \
    --samd_len_threshold 5 \
    --samd_len_bias 5 \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max_cache_len ${MAX_CACHE_LEN} \
    --question-end ${SMOKE_N} \
    --collect_diagnosis_trace

# ---- Step 3 smoke: SAM-only on SMOKE_N questions (no trace flag) ----
# DynSAM only; no static SAM pkl required.
CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_sam_only \
    --template llama3 \
    --model-type llama3 \
    --bench-name ${BENCH} \
    --model-path ${MODEL_PATH} \
    --model-id smoke_sam_only \
    --samd_max_predicts 70 \
    --samd_len_bias 5 \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --max_cache_len ${MAX_CACHE_LEN} \
    --question-end ${SMOKE_N}

# ---- Step 6: aggregate the three smoke answer files ----
python -m evaluation.analyze_vmiss \
    --pure_eagle3 evaluation/data/${BENCH}/model_answer/smoke_pure_eagle3.jsonl \
    --samd_eagle3 evaluation/data/${BENCH}/model_answer/smoke_samd_eagle3.jsonl \
    --sam_only    evaluation/data/${BENCH}/model_answer/smoke_sam_only.jsonl

echo
echo "=========================================="
echo "smoke PASS. Full 80-question benchmark:"
echo "  bash scripts/run_medqa_vmiss_eval.sh"
echo "=========================================="
