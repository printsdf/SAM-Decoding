#!/bin/bash
# Phase 2: Boundary Graft evaluation on HumanEval

set -e

# Default parameters
THRESHOLD=${1:-1.5}
MAX_SAM_NODES=${2:-8}
MIN_DEPTH=${3:-3}
MAX_DEPTH=${4:-8}
FULL=${5:-false}

# Question range
if [ "$FULL" = "true" ]; then
    Q_BEGIN=0
    Q_END=164
    OUTPUT_SUFFIX="full"
else
    Q_BEGIN=0
    Q_END=20
    OUTPUT_SUFFIX="dev20"
fi

OUTPUT_DIR="outputs/phase2_boundary_graft"
mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE="${OUTPUT_DIR}/humaneval_${OUTPUT_SUFFIX}_t${THRESHOLD}_n${MAX_SAM_NODES}.jsonl"

echo "=========================================="
echo "Phase 2: Boundary Graft Evaluation"
echo "=========================================="
echo "Threshold: $THRESHOLD"
echo "Max SAM nodes: $MAX_SAM_NODES"
echo "Depth range: [$MIN_DEPTH, $MAX_DEPTH]"
echo "Questions: $Q_BEGIN-$Q_END"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

python evaluation/inference_samd.py \
    --model-path /mnt/hwfile/medai/models/Meta-Llama-3.1-8B-Instruct \
    --tree-model-path /mnt/hwfile/medai/models/EAGLE3-Llama-3.1-8B-Instruct \
    --dataset humaneval \
    --q-begin "$Q_BEGIN" \
    --q-end "$Q_END" \
    --tree-method eagle3 \
    --fusion-mode boundary_graft \
    --boundary-graft-threshold "$THRESHOLD" \
    --boundary-graft-max-sam-nodes "$MAX_SAM_NODES" \
    --boundary-graft-min-depth "$MIN_DEPTH" \
    --boundary-graft-max-depth "$MAX_DEPTH" \
    --collect-diagnosis-trace \
    --output "$OUTPUT_FILE"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Output saved to: $OUTPUT_FILE"
echo "=========================================="
