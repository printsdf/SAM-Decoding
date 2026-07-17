#!/bin/bash
# Oracle 上限：无条件在贪心 top-path 每点 + 叶挂满 SAM（n_predicts），
# verifier 自动选最长接受路径 → 当前机制集合 + 完美选择 + 无限预算的 MAT 天花板。
#
# 判读：oracle_MAT 对 comp_r8k2_e16(+6.28%) 的差 = 选择器/预算还能榨多少。
#   差小 → SAM 源到顶（转 B）；差大 → 选择器欠佳（上 C 学习 policy）。
# baseline / comp_r8k2_e16 复用已有结果。oracle 树很大，只跑 MAT，tok/s 不作数。
#
# 用法:
#   nohup bash scripts/run_drafter_mars_oracle_batch.sh > oracle_batch.log 2>&1 &

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ==================== 配置 ====================
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B}
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-164}
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
if ! python3 -m pytest tests/test_drafter_mars_adaptive.py tests/test_drafter_mars_gate.py tests/test_subtree_graft.py tests/test_eagle3_parents.py -q; then
  notify_feishu "Oracle batch: 测试失败，已中止。host: $(hostname)"
  do_shutdown
  exit 1
fi

COMMON_ARGS=(
  --model-type llama3 --template llama3
  --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH"
  --bench-name humaneval --question-begin "$QUESTION_BEGIN" --question-end "$QUESTION_END"
  --tree_method eagle3 --samd_len_threshold 5 --samd_len_bias 5 --max_cache_len 4096
  --fusion_mode drafter_mars --drafter_mars_theta 0.86
)

FAILED=""
run_one() {
  local model_id=$1; shift
  echo "==== $(date '+%F %T') START ${model_id} ===="
  if ! python3 evaluation/inference_samd.py --model-id "$model_id" "${COMMON_ARGS[@]}" "$@"; then
    echo "==== FAILED ${model_id} ===="; FAILED="${FAILED} ${model_id}"
  fi
  echo "==== $(date '+%F %T') DONE ${model_id} ===="
}

# oracle 上限（无限预算，只看 MAT）
run_one oracle_upper --drafter_mars_oracle

echo "==== $(date '+%F %T') SUMMARY ===="
python3 scripts/speed.py \
  "${ANSWER_DIR}/comp_baseline.jsonl" \
  "${ANSWER_DIR}/comp_r8k2_e16.jsonl" \
  "${ANSWER_DIR}/oracle_upper.jsonl" \
  | tee "${ANSWER_DIR}/oracle_batch_summary.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

notify_feishu "Drafter-MARS oracle 上限 完成 $(date '+%F %T')
host: $(hostname)
FAILED:${FAILED:-" 无"}

$(head -10 "${ANSWER_DIR}/oracle_batch_summary.txt" 2>/dev/null || echo "summary 缺失")"

do_shutdown
