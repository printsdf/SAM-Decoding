#!/bin/bash
# Phase A 自适应批次：ref 等价性 + 自适应 theta + 动态 budget + 多点 graft + trio
#
# 用法:
#   nohup bash scripts/run_drafter_mars_adaptive_batch.sh > adaptive_batch.log 2>&1 &
#   tail -f adaptive_batch.log
#
# ref 臂即 default-off 等价性检查（MAT 必须 = 7.7849）。SHUTDOWN=0 禁用关机。

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ==================== 配置 ====================
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B}
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-164}
THETA=${THETA:-0.86}
EXPECTED_MAT=${EXPECTED_MAT:-7.7849}
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
  notify_feishu "Adaptive batch: 测试失败，已中止（未跑 eval）。host: $(hostname)"
  do_shutdown
  exit 1
fi

COMMON_ARGS=(
  --model-type llama3 --template llama3
  --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH"
  --bench-name humaneval --question-begin "$QUESTION_BEGIN" --question-end "$QUESTION_END"
  --tree_method eagle3 --samd_len_threshold 5 --samd_len_bias 5 --max_cache_len 4096
  --fusion_mode drafter_mars --drafter_mars_repair graft --drafter_mars_theta "$THETA"
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

# 1) default-off 等价性参照
run_one adaptA_ref

# 2) 自适应 theta（目标触发率 0.75 / 0.60）
run_one adaptA_theta075 --drafter_mars_adaptive_theta --drafter_mars_target_trigger_rate 0.75
run_one adaptA_theta060 --drafter_mars_adaptive_theta --drafter_mars_target_trigger_rate 0.60

# 3) 动态 budget
run_one adaptA_budget_depth --drafter_mars_budget_mode depth
run_one adaptA_budget_ratio --drafter_mars_budget_mode ratio

# 4) 多点 graft
run_one adaptA_k2 --drafter_mars_max_grafts 2
run_one adaptA_k3 --drafter_mars_max_grafts 3

# 5) trio
run_one adaptA_trio --drafter_mars_adaptive_theta --drafter_mars_target_trigger_rate 0.75 \
  --drafter_mars_budget_mode depth --drafter_mars_max_grafts 2

# 6) 等价性断言 + 汇总
echo "==== $(date '+%F %T') SUMMARY ===="
MAT_CHECK=$(python3 - <<PY
import json
tok = steps = 0
for line in open("${ANSWER_DIR}/adaptA_ref.jsonl"):
    c = json.loads(line)["choices"][0]
    tok += sum(c["new_tokens"]); steps += sum(c["decoding_steps"])
mat = tok / steps if steps else 0.0
print(f"{'OK' if f'{mat:.4f}' == '${EXPECTED_MAT}' else 'REGRESSION'} MAT={mat:.4f} expected=${EXPECTED_MAT}")
PY
) || MAT_CHECK="MAT check failed to run"
echo "equivalence: ${MAT_CHECK}"

python3 scripts/speed.py \
  "${ANSWER_DIR}/adaptA_ref.jsonl" \
  "${ANSWER_DIR}/adaptA_theta075.jsonl" \
  "${ANSWER_DIR}/adaptA_theta060.jsonl" \
  "${ANSWER_DIR}/adaptA_budget_depth.jsonl" \
  "${ANSWER_DIR}/adaptA_budget_ratio.jsonl" \
  "${ANSWER_DIR}/adaptA_k2.jsonl" \
  "${ANSWER_DIR}/adaptA_k3.jsonl" \
  "${ANSWER_DIR}/adaptA_trio.jsonl" \
  | tee "${ANSWER_DIR}/adaptive_batch_summary.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

# 7) 飞书 + 关机
notify_feishu "Drafter-MARS adaptive batch 完成 $(date '+%F %T')
host: $(hostname)
equivalence: ${MAT_CHECK}
FAILED:${FAILED:-" 无"}

$(head -20 "${ANSWER_DIR}/adaptive_batch_summary.txt" 2>/dev/null || echo "summary 缺失")"

do_shutdown
