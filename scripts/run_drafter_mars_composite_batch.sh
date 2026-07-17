#!/bin/bash
# 合成最优批次：sequence 保持作者 40 视野，修补/延伸各用机制级视野
#
# 跨批结论：sequence=40 不动；repair 甜点 ~8（小而准，K2 有效）；
# extension 甜点 ~16-24（单位节点价值最高）。本批合成并定最终操作点。
#
# 用法:
#   nohup bash scripts/run_drafter_mars_composite_batch.sh > composite_batch.log 2>&1 &

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ==================== 配置 ====================
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B}
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-164}
THETA=${THETA:-0.86}
SHUTDOWN=${SHUTDOWN:-1}
ANSWER_DIR=evaluation/data/humaneval/model_answer
FEISHU_WEBHOOK=${FEISHU_WEBHOOK:-}
# ==================== 配置结束 ====================

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

notify_feishu() {
  [ -z "$FEISHU_WEBHOOK" ] && return 0
  FEISHU_WEBHOOK="$FEISHU_WEBHOOK" TEXT="$1" python3 - <<'PY' || echo "feishu notify failed"
import json, os, urllib.request
req = urllib.request.Request(
    os.environ["FEISHU_WEBHOOK"],
    data=json.dumps({"msg_type": "text", "content": {"text": os.environ["TEXT"]}}).encode(),
    headers={"Content-Type": "application/json"},
)
print("feishu:", urllib.request.urlopen(req, timeout=10).read().decode())
PY
}

do_shutdown() {
  if [ "$SHUTDOWN" = "1" ]; then
    echo "$(date '+%F %T') 正在关机..."
    /usr/bin/shutdown -h now
  else
    echo "SHUTDOWN=0，跳过关机"
  fi
}

echo "==== $(date '+%F %T') TESTS ===="
if ! python3 -m pytest tests/test_drafter_mars_adaptive.py tests/test_drafter_mars_gate.py tests/test_eagle3_parents.py -q; then
  notify_feishu "Composite batch: 测试失败，已中止。host: $(hostname)"
  do_shutdown
  exit 1
fi

COMMON_ARGS=(
  --model-type llama3 --template llama3
  --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH"
  --bench-name humaneval --question-begin "$QUESTION_BEGIN" --question-end "$QUESTION_END"
  --tree_method eagle3 --samd_len_threshold 5 --samd_len_bias 5 --max_cache_len 4096
)
MARS_ARGS=(--fusion_mode drafter_mars --drafter_mars_repair graft --drafter_mars_theta "$THETA" --drafter_mars_extend)

FAILED=""

run_one() {
  local model_id=$1; shift
  echo "==== $(date '+%F %T') START ${model_id} ===="
  if ! python3 evaluation/inference_samd.py --model-id "$model_id" "${COMMON_ARGS[@]}" "$@"; then
    echo "==== FAILED ${model_id} ===="
    FAILED="${FAILED} ${model_id}"
  fi
  echo "==== $(date '+%F %T') DONE ${model_id} ===="
}

# 1) 同批基线
run_one comp_baseline --fusion_mode none --tree_fusion none

# 2) 合成臂：sequence=40 固定，机制级视野组合
run_one comp_r8_e16      "${MARS_ARGS[@]}" --drafter_mars_graft_horizon 8  --drafter_mars_extend_horizon 16
run_one comp_r8_e24      "${MARS_ARGS[@]}" --drafter_mars_graft_horizon 8  --drafter_mars_extend_horizon 24
run_one comp_r8k2_e16    "${MARS_ARGS[@]}" --drafter_mars_graft_horizon 8  --drafter_mars_extend_horizon 16 --drafter_mars_max_grafts 2
run_one comp_r8k2_e24    "${MARS_ARGS[@]}" --drafter_mars_graft_horizon 8  --drafter_mars_extend_horizon 24 --drafter_mars_max_grafts 2
run_one comp_r12_e24     "${MARS_ARGS[@]}" --drafter_mars_graft_horizon 12 --drafter_mars_extend_horizon 24

# 3) 汇总
echo "==== $(date '+%F %T') SUMMARY ===="
python3 scripts/speed.py \
  "${ANSWER_DIR}/comp_baseline.jsonl" \
  "${ANSWER_DIR}/comp_r8_e16.jsonl" \
  "${ANSWER_DIR}/comp_r8_e24.jsonl" \
  "${ANSWER_DIR}/comp_r8k2_e16.jsonl" \
  "${ANSWER_DIR}/comp_r8k2_e24.jsonl" \
  "${ANSWER_DIR}/comp_r12_e24.jsonl" \
  | tee "${ANSWER_DIR}/composite_batch_summary.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

# 4) 飞书 + 关机
notify_feishu "Drafter-MARS composite batch 完成 $(date '+%F %T')
host: $(hostname)
FAILED:${FAILED:-" 无"}

$(head -20 "${ANSWER_DIR}/composite_batch_summary.txt" 2>/dev/null || echo "summary 缺失")"

do_shutdown
