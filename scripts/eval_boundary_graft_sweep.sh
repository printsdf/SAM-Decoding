#!/bin/bash
# Phase 2: Boundary Graft threshold sweep

set -e

OUTPUT_DIR="outputs/phase2_boundary_graft"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Phase 2: Boundary Graft Threshold Sweep"
echo "=========================================="
echo "Running 6 threshold configurations..."
echo "Dataset: HumanEval q0-19 (dev set)"
echo "=========================================="

THRESHOLDS=(0.5 1.0 1.5 2.0 2.5 3.0)

for threshold in "${THRESHOLDS[@]}"; do
    echo ""
    echo ">>> Running threshold=$threshold"
    bash scripts/eval_boundary_graft.sh "$threshold" 8 3 8 false
    echo "<<< Completed threshold=$threshold"
done

echo ""
echo "=========================================="
echo "Threshold sweep complete!"
echo "=========================================="
echo ""
echo "Results summary:"
for threshold in "${THRESHOLDS[@]}"; do
    output_file="${OUTPUT_DIR}/humaneval_dev20_t${threshold}_n8.jsonl"
    if [ -f "$output_file" ]; then
        echo "  threshold=$threshold: $output_file"
    fi
done

echo ""
echo "Next step: Analyze results and select best threshold"
