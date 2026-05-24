#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=0

CUDA_VISIBLE_DEVICES=${devices} \
    python -m tests.test_samd \
    --model_path /root/Models/Meta-Llama-3.1-8B-Instruct \
    --device "cuda" \
    --tree_method eagle3 \
    --tree_model_path /root/Models/EAGLE3-LLaMA3.1-Instruct-8B
