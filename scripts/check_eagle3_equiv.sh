#!/bin/bash
# S2 验收脚本：跑 base greedy 和 samd[eagle3]，逐 token diff。
# design 第 3 节场景 2 的金标准对照。
set -e
set -x

cd $(dirname $0)/..

devices=0

CUDA_VISIBLE_DEVICES=${devices} \
    python -m tests.check_eagle3_equiv \
    --model_path /root/Models/Meta-Llama-3.1-8B-Instruct \
    --tree_model_path /root/Models/EAGLE3-LLaMA3.1-Instruct-8B \
    --tree_method eagle3 \
    --max_new_tokens 64 \
    --device "cuda"
