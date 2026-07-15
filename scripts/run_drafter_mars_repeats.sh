#!/bin/bash
# 收尾批次：theta=0.86 重复验证 + always-on 近似对照 + 自动关机/飞书
#
# 用法:
#   nohup bash scripts/run_drafter_mars_repeats.sh > repeats.log 2>&1 &
#   tail -f repeats.log
#
# 交错运行 baseline / t086 以消除时段漂移；SHUTDOWN=0 可禁用关机。

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ==================== 配置 ====================
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B}
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-164}
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

COMMON_ARGS=(
  --model-type llama3 --template llama3
  --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH"
  --bench-name humaneval --question-begin "$QUESTION_BEGIN" --question-end "$QUESTION_END"
  --tree_method eagle3 --samd_len_threshold 5 --samd_len_bias 5 --max_cache_len 4096
)

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

run_base() { run_one "$1" --fusion_mode none --tree_fusion none; }
run_mars() { run_one "$1" --fusion_mode drafter_mars --drafter_mars_repair graft --drafter_mars_theta "$2"; }

# 1) 交错重复：t086 x3 + baseline x2（原始各一次已在库）
run_mars samd_eagle3_drafter_mars_graft_t086_rep1 0.86
run_base samd_eagle3_baseline_rep1
run_mars samd_eagle3_drafter_mars_graft_t086_rep2 0.86
run_base samd_eagle3_baseline_rep2
run_mars samd_eagle3_drafter_mars_graft_t086_rep3 0.86

# 2) always-on 近似对照（theta=0.5 触发率趋近 100%，earliest parent 变浅）
run_mars samd_eagle3_drafter_mars_graft_t050_alwayson 0.50

# 3) 汇总（第一列为原始 baseline 参照）
echo "==== $(date '+%F %T') SUMMARY ===="
python3 scripts/speed.py \
  "${ANSWER_DIR}/samd_eagle3_baseline.jsonl" \
  "${ANSWER_DIR}"/samd_eagle3_baseline_rep*.jsonl \
  "${ANSWER_DIR}/samd_eagle3_drafter_mars_graft_t086.jsonl" \
  "${ANSWER_DIR}"/samd_eagle3_drafter_mars_graft_t086_rep*.jsonl \
  "${ANSWER_DIR}/samd_eagle3_drafter_mars_graft_t050_alwayson.jsonl" \
  | tee "${ANSWER_DIR}/repeats_summary.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

# 4) 飞书 + 关机
notify_feishu "Drafter-MARS repeats 完成 $(date '+%F %T')
host: $(hostname)
FAILED:${FAILED:-" 无"}

$(head -20 "${ANSWER_DIR}/repeats_summary.txt" 2>/dev/null || echo "repeats_summary.txt 缺失")"

if [ "$SHUTDOWN" = "1" ]; then
  echo "$(date '+%F %T') 全部结束，正在关机..."
  /usr/bin/shutdown -h now
else
  echo "SHUTDOWN=0，跳过关机"
fi
