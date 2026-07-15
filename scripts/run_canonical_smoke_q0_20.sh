#!/usr/bin/env bash
# Re-run q0-20 smoke with canonical HumanEval question file.
# This verifies that EAGLE MAT and oracle ceiling return to expected ranges
# after fixing the question file provenance issue.

set -euo pipefail

cd "$(dirname "$0")/.."

# Smoke configuration: q0-20, train on q0-10, validate on q10-20
export QUESTION_BEGIN=0
export QUESTION_END=20
export TRAIN_BEGIN=0
export TRAIN_END=10
export VALID_BEGIN=10
export VALID_END=20

# Output to a new directory to avoid overwriting the old smoke
export MODEL_ID=drafter_mars_canonical_smoke_q0_20
export PROFILE_OUTPUT_DIR=evaluation/data/humaneval/canonical_smoke

# Use Drafter-MARS predictor suite
export BOUNDARY_PREDICTOR_SUITE=drafter_mars

# Standard model paths (adjust if needed)
export MODEL_PATH=${MODEL_PATH:-/root/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct}
export TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B}
export SAM_PATH=${SAM_PATH:-}

# Enable boundary analyzer
export RUN_BOUNDARY_ANALYZER=1
export BOUNDARY_MAX_THRESHOLDS=256
export BOUNDARY_NODE_BUDGET=60

echo "=========================================="
echo "HumanEval Canonical Q0-20 Smoke"
echo "=========================================="
echo "Question file: evaluation/data/humaneval/question.jsonl"
echo "Question file SHA256: fc49f9304222ac25c3ff5c4ede32e2d61646bd08307da80f0ad806ee60952407"
echo "Question range: ${QUESTION_BEGIN}-${QUESTION_END}"
echo "Train range: ${TRAIN_BEGIN}-${TRAIN_END}"
echo "Valid range: ${VALID_BEGIN}-${VALID_END}"
echo "Output directory: ${PROFILE_OUTPUT_DIR}"
echo "Model ID: ${MODEL_ID}"
echo "Predictor suite: ${BOUNDARY_PREDICTOR_SUITE}"
echo "=========================================="
echo ""

# Verify question file SHA256
actual_sha=$(shasum -a 256 evaluation/data/humaneval/question.jsonl | awk '{print $1}')
expected_sha="fc49f9304222ac25c3ff5c4ede32e2d61646bd08307da80f0ad806ee60952407"

if [ "${actual_sha}" != "${expected_sha}" ]; then
    echo "ERROR: Question file SHA256 mismatch!"
    echo "Expected: ${expected_sha}"
    echo "Actual:   ${actual_sha}"
    echo ""
    echo "Regenerate the canonical question file with:"
    echo "  python evaluation/humaneval_prep.py"
    exit 1
fi

echo "✓ Question file SHA256 verified: ${actual_sha}"
echo ""

# Run the boundary predictor profile script
bash scripts/profile_boundary_predictor_humaneval.sh

echo ""
echo "=========================================="
echo "Smoke complete. Check results at:"
echo "  ${PROFILE_OUTPUT_DIR}/${MODEL_ID}.oracle_results.json"
echo "  ${PROFILE_OUTPUT_DIR}/${MODEL_ID}.oracle_rejection_boundary.json"
echo ""
echo "Expected outcomes (if question file is consistent with original):"
echo "  - EAGLE-only MAT: ~3.5-4.0 (down from 6.79 in old smoke)"
echo "  - Same-trace oracle ceiling: >= +7% (up from +3.91% in old smoke)"
echo "=========================================="
