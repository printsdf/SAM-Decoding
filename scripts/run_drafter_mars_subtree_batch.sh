#!/bin/bash
# SAM 子树注入 vs 直链修补：同预算下对打
#
# 假设：在不确定父节点注入一棵小 SAM 子树（多分叉候选）比单条直链信息量更高，
# verifier 有多条可接受路径。子树用 dyn SAM 的 cnt_endpos 频次 best-first 展开。
# baseline / comp_r8k2_e16 复用已有结果（用户决定，不重跑）。
#
# 用法:
#   nohup bash scripts/run_drafter_mars_subtree_batch.sh > subtree_batch.log 2>&1 &

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

echo "==== $(date '+%F %T') TESTS ===="
if ! python3 -m pytest tests/test_subtree_graft.py tests/test_drafter_mars_adaptive.py tests/test_drafter_mars_gate.py tests/test_eagle3_parents.py -q; then
  notify_feishu "Subtree batch: 测试失败，已中止。host: $(hostname)"
  do_shutdown
  exit 1
fi

COMMON_ARGS=(
  --model-type llama3 --template llama3
  --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH"
  --bench-name humaneval --question-begin "$QUESTION_BEGIN" --question-end "$QUESTION_END"
  --tree_method eagle3 --samd_len_threshold 5 --samd_len_bias 5 --max_cache_len 4096
  --fusion_mode drafter_mars --drafter_mars_theta "$THETA" --drafter_mars_extend --drafter_mars_extend_horizon 16
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

# 子树注入臂：graft_horizon = 子树 max_nodes（同直链预算），top_k 控分叉宽度
run_one sub_g8_k4  --drafter_mars_repair subtree --drafter_mars_graft_horizon 8  --sam_tree_top_k 4 --sam_tree_max_depth 6
run_one sub_g12_k4 --drafter_mars_repair subtree --drafter_mars_graft_horizon 12 --sam_tree_top_k 4 --sam_tree_max_depth 6
run_one sub_g12_k2 --drafter_mars_repair subtree --drafter_mars_graft_horizon 12 --sam_tree_top_k 2 --sam_tree_max_depth 6
run_one sub_g16_k4 --drafter_mars_repair subtree --drafter_mars_graft_horizon 16 --sam_tree_top_k 4 --sam_tree_max_depth 8

# 汇总（第一列复用已有 baseline + 直链最优对照）
echo "==== $(date '+%F %T') SUMMARY ===="
python3 scripts/speed.py \
  "${ANSWER_DIR}/comp_baseline.jsonl" \
  "${ANSWER_DIR}/comp_r8k2_e16.jsonl" \
  "${ANSWER_DIR}/sub_g8_k4.jsonl" \
  "${ANSWER_DIR}/sub_g12_k4.jsonl" \
  "${ANSWER_DIR}/sub_g12_k2.jsonl" \
  "${ANSWER_DIR}/sub_g16_k4.jsonl" \
  | tee "${ANSWER_DIR}/subtree_batch_summary.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

notify_feishu "Drafter-MARS subtree batch 完成 $(date '+%F %T')
host: $(hostname)
FAILED:${FAILED:-" 无"}

$(head -20 "${ANSWER_DIR}/subtree_batch_summary.txt" 2>/dev/null || echo "summary 缺失")"

do_shutdown
