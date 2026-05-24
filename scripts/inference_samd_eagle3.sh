#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

# Meta-Llama-3.1-8B-Instruct + EAGLE3 (SAM[EAGLE3] hybrid)
#
# Dataset: put your benchmark questions at
#   evaluation/data/${bench_name}/question.jsonl
# (each line is a JSON object with at least: question_id, category, turns)
# Outputs land at
#   evaluation/data/${bench_name}/model_answer/${model_id}.jsonl
#
# To switch to your vertical dataset, change --bench-name below.

CUDA_VISIBLE_DEVICES=${devices} \
    python -m evaluation.inference_samd \
    --template llama3 \
    --model-type llama3 \
    --bench-name spec_bench \
    --model-path /root/Models/Meta-Llama-3.1-8B-Instruct \
    --model-id meta-llama-3.1-8b-instruct-samd-eagle3 \
    --sam_path local_cache/sam_alpaca_vicuna-7b-v1.3_min-endpos.pkl \
    --tree_method eagle3 \
    --tree_model_path /root/Models/EAGLE3-LLaMA3.1-Instruct-8B \
    --samd_n_predicts 40 \
    --samd_len_threshold 5 \
    --samd_len_bias 5
