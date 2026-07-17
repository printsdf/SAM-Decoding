#!/bin/bash
# Drafter-MARS theta sweep + profile + 自动关机
#
# 用法（服务器，后台运行）:
#   nohup bash scripts/run_drafter_mars_sweep.sh > sweep.log 2>&1 &
#   tail -f sweep.log
#
# 跑完自动 shutdown；调试时 SHUTDOWN=0 nohup bash ... 可禁用关机。

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ==================== 配置 ====================
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B}
THETAS=${THETAS:-"0.88 0.94 0.98"}   # 0.90 已有结果，不重跑
PROFILE_THETA=${PROFILE_THETA:-0.90}
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-164}
SHUTDOWN=${SHUTDOWN:-1}
ANSWER_DIR=evaluation/data/humaneval/model_answer
BASELINE_ID=${BASELINE_ID:-samd_eagle3_baseline}
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

COMMON_ARGS=(
  --model-type llama3 --template llama3
  --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH"
  --bench-name humaneval --question-begin "$QUESTION_BEGIN" --question-end "$QUESTION_END"
  --tree_method eagle3 --fusion_mode drafter_mars --drafter_mars_repair graft
  --samd_len_threshold 5 --samd_len_bias 5 --max_cache_len 4096
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

# 1) theta sweep（正式对比，profile 关闭）
for T in $THETAS; do
  run_one "samd_eagle3_drafter_mars_graft_t${T//./}" --drafter_mars_theta "$T"
done

# 2) profile run（仅诊断触发率/分段耗时，tok/s 不作数）
run_one "samd_eagle3_drafter_mars_graft_t${PROFILE_THETA//./}_profile" \
  --drafter_mars_theta "$PROFILE_THETA" --profile-fusion

# 3) 汇总对比（排除 profile run）
echo "==== $(date '+%F %T') SUMMARY ===="
FILES="${ANSWER_DIR}/${BASELINE_ID}.jsonl"
for f in "${ANSWER_DIR}"/samd_eagle3_drafter_mars_graft_t*.jsonl; do
  [[ "$f" == *_profile.jsonl ]] && continue
  FILES="$FILES $f"
done
python3 scripts/speed.py $FILES | tee "${ANSWER_DIR}/sweep_summary.txt"
echo "profile 输出: ${ANSWER_DIR}/samd_eagle3_drafter_mars_graft_t${PROFILE_THETA//./}_profile.jsonl.fusion_profile.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

# 4) 飞书提醒
notify_feishu "Drafter-MARS sweep 完成 $(date '+%F %T')
host: $(hostname)
FAILED:${FAILED:-" 无"}

$(head -20 "${ANSWER_DIR}/sweep_summary.txt" 2>/dev/null || echo "sweep_summary.txt 缺失")"

# 5) 关机
if [ "$SHUTDOWN" = "1" ]; then
  echo "$(date '+%F %T') 全部结束，正在关机..."
  /usr/bin/shutdown -h now
else
  echo "SHUTDOWN=0，跳过关机"
fi
