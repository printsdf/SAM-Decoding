#!/bin/bash
# Fixed-budget boundary repair: prune low-value EAGLE leaves after SAM grafts
# so every fused verifier tree remains root-inclusive <= TREE_BUDGET nodes.
#
# Remote usage:
#   nohup bash scripts/run_drafter_mars_fixed_budget_batch.sh > fixed_budget.log 2>&1 &

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B}
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-164}
THETA=${THETA:-0.86}
TREE_BUDGET=${TREE_BUDGET:-60}
SHUTDOWN=${SHUTDOWN:-0}
ANSWER_DIR=${ANSWER_DIR:-evaluation/data/humaneval/model_answer}
FEISHU_WEBHOOK=${FEISHU_WEBHOOK:-}

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

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
    /usr/bin/shutdown -h now
  fi
}

if ! python3 -m pytest \
  tests/test_subtree_graft.py \
  tests/test_drafter_mars_adaptive.py \
  tests/test_drafter_mars_gate.py \
  tests/test_eagle3_parents.py -q; then
  notify_feishu "Drafter-MARS fixed-budget batch: tests failed on $(hostname)"
  do_shutdown
  exit 1
fi

COMMON_ARGS=(
  --model-type llama3 --template llama3
  --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH"
  --bench-name humaneval --question-begin "$QUESTION_BEGIN" --question-end "$QUESTION_END"
  --tree_method eagle3 --eagle3_total_token 60
  --samd_len_threshold 5 --samd_len_bias 5 --max_cache_len 4096
)
MARS_ARGS=(
  --fusion_mode drafter_mars --drafter_mars_repair graft
  --drafter_mars_theta "$THETA" --drafter_mars_extend
  --drafter_mars_graft_horizon 8 --drafter_mars_extend_horizon 16
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
    FAILED="${FAILED} ${model_id}"
  fi
  echo "==== $(date '+%F %T') DONE ${model_id} ===="
}

run_one budget_baseline --fusion_mode none --tree_fusion none
run_one budget_append_r8_e16 "${MARS_ARGS[@]}"
run_one budget60_r8_e16 "${MARS_ARGS[@]}" \
  --drafter_mars_tree_budget "$TREE_BUDGET"
run_one budget60_r8k2_e16 "${MARS_ARGS[@]}" \
  --drafter_mars_max_grafts 2 \
  --drafter_mars_tree_budget "$TREE_BUDGET"
run_one budget60_r8_e16_profile "${MARS_ARGS[@]}" \
  --drafter_mars_tree_budget "$TREE_BUDGET" \
  --profile-fusion

python3 scripts/speed.py \
  "${ANSWER_DIR}/budget_baseline.jsonl" \
  "${ANSWER_DIR}/budget_append_r8_e16.jsonl" \
  "${ANSWER_DIR}/budget60_r8_e16.jsonl" \
  "${ANSWER_DIR}/budget60_r8k2_e16.jsonl" \
  | tee "${ANSWER_DIR}/fixed_budget_summary.txt"

notify_feishu "Drafter-MARS fixed-budget batch complete on $(hostname)
budget=${TREE_BUDGET}; failed=${FAILED:-none}

$(head -20 "${ANSWER_DIR}/fixed_budget_summary.txt" 2>/dev/null || true)"
do_shutdown
