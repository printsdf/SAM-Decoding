#!/bin/bash
# Preallocated 60-node budget batch.
#
# Instead of generating 60 EAGLE nodes and pruning after grafting, reserve
# seven slots before verification: EAGLE generates its native top-53 tree and
# an r8 SAM repair/extension can add at most seven non-root nodes. The budget-60
# guard remains enabled as an invariant check.
#
# Usage:
#   nohup bash scripts/run_drafter_mars_preallocated_budget_batch.sh > prealloc_batch.log 2>&1 &
#   tail -f prealloc_batch.log

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ==================== Configuration ====================
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B}
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-164}
THETA=${THETA:-0.86}
TREE_BUDGET=${TREE_BUDGET:-60}
SHUTDOWN=${SHUTDOWN:-1}
ANSWER_DIR=${ANSWER_DIR:-evaluation/data/humaneval/model_answer}
FEISHU_WEBHOOK=${FEISHU_WEBHOOK:-}
# ==================== Configuration end ====================

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$ANSWER_DIR"

notify_feishu() {
  [ -z "$FEISHU_WEBHOOK" ] && return 0
  FEISHU_WEBHOOK="$FEISHU_WEBHOOK" TEXT="$1" python3 - <<'PY' || echo "feishu notify failed"
import json
import os
import urllib.request

request = urllib.request.Request(
    os.environ["FEISHU_WEBHOOK"],
    data=json.dumps(
        {"msg_type": "text", "content": {"text": os.environ["TEXT"]}}
    ).encode(),
    headers={"Content-Type": "application/json"},
)
print("feishu:", urllib.request.urlopen(request, timeout=10).read().decode())
PY
}

do_shutdown() {
  if [ "$SHUTDOWN" = "1" ]; then
    echo "$(date '+%F %T') shutting down..."
    /usr/bin/shutdown -h now
  else
    echo "SHUTDOWN=0; skip shutdown"
  fi
}

echo "==== $(date '+%F %T') TESTS ===="
if ! python3 -m pytest \
  tests/test_subtree_graft.py \
  tests/test_drafter_mars_adaptive.py \
  tests/test_drafter_mars_gate.py \
  tests/test_eagle3_parents.py -q; then
  notify_feishu "Drafter-MARS preallocated-budget batch: tests failed on $(hostname)"
  do_shutdown
  exit 1
fi

COMMON_ARGS=(
  --model-type llama3 --template llama3
  --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH"
  --bench-name humaneval --question-begin "$QUESTION_BEGIN" --question-end "$QUESTION_END"
  --tree_method eagle3 --samd_len_threshold 5 --samd_len_bias 5 --max_cache_len 4096
)
MARS_ARGS=(
  --fusion_mode drafter_mars
  --drafter_mars_repair graft
  --drafter_mars_theta "$THETA"
  --drafter_mars_graft_horizon 8
)

FAILED=""

run_one() {
  local model_id=$1
  shift
  echo "==== $(date '+%F %T') START ${model_id} ===="
  if ! python3 evaluation/inference_samd.py \
    --model-id "$model_id" \
    --answer-file "${ANSWER_DIR}/${model_id}.jsonl" \
    "${COMMON_ARGS[@]}" "$@"; then
    echo "==== FAILED ${model_id} ===="
    FAILED="${FAILED} ${model_id}"
  fi
  echo "==== $(date '+%F %T') DONE ${model_id} ===="
}

# 1) Same-boot stock EAGLE3 baseline.
run_one prealloc_baseline \
  --fusion_mode none --tree_fusion none --eagle3_total_token 60

# 2) Current append-only operating point for a same-boot reference.
run_one prealloc_append_r8_e16 \
  "${MARS_ARGS[@]}" \
  --eagle3_total_token 60 \
  --drafter_mars_extend \
  --drafter_mars_extend_horizon 16

# 3) Preallocate seven verifier slots to repair only.
run_one prealloc_e53_r8 \
  "${MARS_ARGS[@]}" \
  --eagle3_total_token 53 \
  --drafter_mars_tree_budget "$TREE_BUDGET"

# 4) Preallocate the same seven slots for either repair or short extension.
run_one prealloc_e53_r8_e8 \
  "${MARS_ARGS[@]}" \
  --eagle3_total_token 53 \
  --drafter_mars_extend \
  --drafter_mars_extend_horizon 8 \
  --drafter_mars_tree_budget "$TREE_BUDGET"

# 5) Profile duplicate: use phase percentages for diagnosis, not tok/s.
run_one prealloc_e53_r8_e8_profile \
  "${MARS_ARGS[@]}" \
  --eagle3_total_token 53 \
  --drafter_mars_extend \
  --drafter_mars_extend_horizon 8 \
  --drafter_mars_tree_budget "$TREE_BUDGET" \
  --profile-fusion

echo "==== $(date '+%F %T') SUMMARY ===="
python3 scripts/speed.py \
  "${ANSWER_DIR}/prealloc_baseline.jsonl" \
  "${ANSWER_DIR}/prealloc_append_r8_e16.jsonl" \
  "${ANSWER_DIR}/prealloc_e53_r8.jsonl" \
  "${ANSWER_DIR}/prealloc_e53_r8_e8.jsonl" \
  | tee "${ANSWER_DIR}/preallocated_budget_summary.txt"

echo "profile output: ${ANSWER_DIR}/prealloc_e53_r8_e8_profile.jsonl.fusion_profile.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

notify_feishu "Drafter-MARS preallocated-budget batch complete $(date '+%F %T')
host: $(hostname)
budget: ${TREE_BUDGET}
failed:${FAILED:- none}

$(head -20 "${ANSWER_DIR}/preallocated_budget_summary.txt" 2>/dev/null || echo "summary missing")"

do_shutdown
