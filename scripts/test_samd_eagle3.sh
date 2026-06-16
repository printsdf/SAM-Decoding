#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

devices=${CUDA_VISIBLE_DEVICES:-0}
model_path=${MODEL_PATH:?MODEL_PATH must be set in .env or environment}
tree_model_path=${TREE_MODEL_PATH:?TREE_MODEL_PATH must be set in .env or environment}
dtype=${DTYPE:-float16}
eagle3_total_token=${EAGLE3_TOTAL_TOKEN:-60}
eagle3_depth=${EAGLE3_DEPTH:-7}
eagle3_top_k=${EAGLE3_TOP_K:-10}

CUDA_VISIBLE_DEVICES=${devices} \
    python -m tests.test_samd \
    --model_path "${model_path}" \
    --device "cuda" \
    --dtype "${dtype}" \
    --tree_method eagle3 \
    --tree_model_path "${tree_model_path}" \
    --eagle3_total_token "${eagle3_total_token}" \
    --eagle3_depth "${eagle3_depth}" \
    --eagle3_top_k "${eagle3_top_k}"
