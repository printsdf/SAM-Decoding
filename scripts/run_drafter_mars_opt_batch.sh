#!/bin/bash
# 优化验证批次：torch 测试 → 基线参照 → t086 优化版等价性验证 → graft 预算扫描
#
# 用法:
#   nohup bash scripts/run_drafter_mars_opt_batch.sh > opt_batch.log 2>&1 &
#   tail -f opt_batch.log
#
# 等价性判据：t086_opt 的 MAT 必须精确等于 7.7849（bit-identical 证明），
# 之后 tok/s 与同批 baseline_opt 对比。SHUTDOWN=0 可禁用关机。

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ==================== 配置 ====================
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B}
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-164}
THETA=${THETA:-0.86}
BUDGETS=${BUDGETS:-"4 6 12"}     # 8 即 t086_opt 本体，不重复
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

# 0) torch 侧等价性测试，失败则中止（不烧后续 GPU 时间）
echo "==== $(date '+%F %T') TESTS ===="
if ! python3 -m pytest tests/test_eagle3_parents.py tests/test_drafter_mars_gate.py -q; then
  notify_feishu "Drafter-MARS opt batch: 等价性测试失败，已中止（未跑 eval）。host: $(hostname)"
  do_shutdown
  exit 1
fi

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

# 1) 同批基线参照（重启后的机器状态与历史批次不可比，必须同批对照）
run_one samd_eagle3_baseline_opt --fusion_mode none --tree_fusion none

# 2) t086 优化版（等价性 + 吞吐主对比）
run_one samd_eagle3_drafter_mars_graft_t086_opt \
  --fusion_mode drafter_mars --drafter_mars_repair graft --drafter_mars_theta "$THETA"

# 3) graft 预算扫描（同 theta）
for B in $BUDGETS; do
  run_one "samd_eagle3_drafter_mars_graft_t086_b${B}" \
    --fusion_mode drafter_mars --drafter_mars_repair graft --drafter_mars_theta "$THETA" \
    --boundary_graft_max_sam_nodes "$B"
done

# 4) 等价性断言 + 汇总
echo "==== $(date '+%F %T') SUMMARY ===="
MAT_CHECK=$(python3 - <<PY
import json
tok = steps = 0
for line in open("${ANSWER_DIR}/samd_eagle3_drafter_mars_graft_t086_opt.jsonl"):
    c = json.loads(line)["choices"][0]
    tok += sum(c["new_tokens"]); steps += sum(c["decoding_steps"])
mat = tok / steps if steps else 0.0
print(f"{'OK' if f'{mat:.4f}' == '${EXPECTED_MAT}' else 'REGRESSION'} MAT={mat:.4f} expected=${EXPECTED_MAT}")
PY
) || MAT_CHECK="MAT check failed to run"
echo "equivalence: ${MAT_CHECK}"

python3 scripts/speed.py \
  "${ANSWER_DIR}/samd_eagle3_baseline_opt.jsonl" \
  "${ANSWER_DIR}/samd_eagle3_drafter_mars_graft_t086_opt.jsonl" \
  "${ANSWER_DIR}"/samd_eagle3_drafter_mars_graft_t086_b*.jsonl \
  | tee "${ANSWER_DIR}/opt_batch_summary.txt"
[ -n "$FAILED" ] && echo "FAILED runs:${FAILED}"

# 5) 飞书 + 关机
notify_feishu "Drafter-MARS opt batch 完成 $(date '+%F %T')
host: $(hostname)
equivalence: ${MAT_CHECK}
FAILED:${FAILED:-" 无"}

$(head -20 "${ANSWER_DIR}/opt_batch_summary.txt" 2>/dev/null || echo "opt_batch_summary.txt 缺失")"

do_shutdown
