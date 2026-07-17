#!/bin/bash
# 容量重分配批次：缩 EAGLE 树换 SAM 延伸，n_predicts 作为唯一长度旋钮（作者语义）
#
# 背景（horizon batch 判决）：verify 对 token 数在 ~80-100 之后 compute-bound，
# 单位节点价值 延伸 > 修补 > EAGLE 深层分支。目标：总树规模 ≈ baseline(61)，
# 用低价值 EAGLE 节点换高价值 SAM 延伸。
#
# 用法:
#   nohup bash scripts/run_drafter_mars_realloc_batch.sh > realloc_batch.log 2>&1 &

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
  notify_feishu "Realloc batch: 测试失败，已中止。host: $(hostname)"
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

# 1) 同批基线（全尺寸 EAGLE 61 节点）
run_one realloc_baseline --fusion_mode none --tree_fusion none

# 2) 不缩树，n_predicts 找延伸长度的甜点（树 ≤ 61+np）
run_one realloc_e60_np16 "${MARS_ARGS[@]}" --samd_n_predicts 16
run_one realloc_e60_np24 "${MARS_ARGS[@]}" --samd_n_predicts 24

# 3) 缩树重分配（总规模 ≈ baseline）
run_one realloc_e45_np16 "${MARS_ARGS[@]}" --eagle3_total_token 45 --samd_n_predicts 16
run_one realloc_e45_np24 "${MARS_ARGS[@]}" --eagle3_total_token 45 --samd_n_predicts 24
run_one realloc_e40_np24 "${MARS_ARGS[@]}" --eagle3_total_token 40 --samd_n_predicts 24

# 4) 缩树对照（无 SAM，隔离缩树本身的损失）
run_one realloc_e45_none --fusion_mode none --tree_fusion none --eagle3_total_token 45

# 5) 汇总
echo "==== $(date '+%F %T') SUMMARY ===="
python3 scripts/speed.py \
  "${ANSWER_DIR}/realloc_baseline.jsonl" \
  "${ANSWER_DIR}/realloc_e60_np16.jsonl" \
  "${ANSWER_DIR}/realloc_e60_np24.jsonl" \
  "${ANSWER_DIR}/realloc_e45_np16.jsonl" \
  "${ANSWER_DIR}/realloc_e45_np24.jsonl" \
  "${ANSWER_DIR}/realloc_e40_np24.jsonl" \
  "${ANSWER_DIR}/realloc_e45_none.jsonl" \
  | tee "${ANSWER_DIR}/realloc_batch_summary.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

# 6) 飞书 + 关机
notify_feishu "Drafter-MARS realloc batch 完成 $(date '+%F %T')
host: $(hostname)
FAILED:${FAILED:-" 无"}

$(head -20 "${ANSWER_DIR}/realloc_batch_summary.txt" 2>/dev/null || echo "summary 缺失")"

do_shutdown
