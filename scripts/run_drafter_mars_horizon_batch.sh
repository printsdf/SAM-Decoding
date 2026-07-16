#!/bin/bash
# Author-horizon 批次：graft/extension 均用 SAM 原生 n_predicts 视野（无人工上限）
#
# 语义变更说明：graft 长度 8 → n_predicts(40)，旧 b8 操作点数字不再可比，
# 全部臂与同批 baseline 对照。
#
# 用法:
#   nohup bash scripts/run_drafter_mars_horizon_batch.sh > horizon_batch.log 2>&1 &
#   tail -f horizon_batch.log

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
FEISHU_WEBHOOK=${FEISHU_WEBHOOK:-https://open.feishu.cn/open-apis/bot/v2/hook/73fdb716-16d7-4948-8e9a-3db134b5d234}
# ==================== 配置结束 ====================

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

notify_feishu() {
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

# 0) torch 侧测试，失败即止
echo "==== $(date '+%F %T') TESTS ===="
if ! python3 -m pytest tests/test_drafter_mars_adaptive.py tests/test_drafter_mars_gate.py tests/test_eagle3_parents.py -q; then
  notify_feishu "Horizon batch: 测试失败，已中止。host: $(hostname)"
  do_shutdown
  exit 1
fi

COMMON_ARGS=(
  --model-type llama3 --template llama3
  --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH"
  --bench-name humaneval --question-begin "$QUESTION_BEGIN" --question-end "$QUESTION_END"
  --tree_method eagle3 --samd_len_threshold 5 --samd_len_bias 5 --max_cache_len 4096
)
MARS_ARGS=(--fusion_mode drafter_mars --drafter_mars_repair graft --drafter_mars_theta "$THETA")

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
run_one horizon_baseline --fusion_mode none --tree_fusion none

# 2) 修补臂（author 视野）
run_one horizon_g40      "${MARS_ARGS[@]}"
run_one horizon_g40_k2   "${MARS_ARGS[@]}" --drafter_mars_max_grafts 2

# 3) 延伸臂
run_one horizon_ext      "${MARS_ARGS[@]}" --drafter_mars_extend
run_one horizon_g40_k2_ext "${MARS_ARGS[@]}" --drafter_mars_max_grafts 2 --drafter_mars_extend

# 4) 汇总
echo "==== $(date '+%F %T') SUMMARY ===="
python3 scripts/speed.py \
  "${ANSWER_DIR}/horizon_baseline.jsonl" \
  "${ANSWER_DIR}/horizon_g40.jsonl" \
  "${ANSWER_DIR}/horizon_g40_k2.jsonl" \
  "${ANSWER_DIR}/horizon_ext.jsonl" \
  "${ANSWER_DIR}/horizon_g40_k2_ext.jsonl" \
  | tee "${ANSWER_DIR}/horizon_batch_summary.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

# 5) 飞书 + 关机
notify_feishu "Drafter-MARS horizon batch 完成 $(date '+%F %T')
host: $(hostname)
FAILED:${FAILED:-" 无"}

$(head -20 "${ANSWER_DIR}/horizon_batch_summary.txt" 2>/dev/null || echo "summary 缺失")"

do_shutdown
