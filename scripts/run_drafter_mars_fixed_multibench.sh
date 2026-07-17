#!/bin/bash
# Fixed-parameter Spec-Bench batch (Phase A operating point).
#
# Dataset: evaluation/data/spec_bench/question.jsonl (480q, 6 paper domains)
#   multi-turn conversation (80) | translation (80) | summarization (80)
#   qa (80) | math_reasoning (80) | rag (80)
#
# Arms (same machine / same session; fair tok/s):
#   1) baseline  — SAM[EAGLE3], fusion off
#   2) k1        — drafter_mars graft, theta=0.86, budget=8, K=1
#   3) k2        — same as k1 but K=2  (new operating point)
#
# Usage (server):
#   nohup bash scripts/run_drafter_mars_fixed_multibench.sh > specbench_fixed.log 2>&1 &
#   tail -f specbench_fixed.log
#
# Overrides:
#   ARMS="baseline k2"              # skip k1
#   QUESTION_BEGIN=0 QUESTION_END=20  # smoke
#   SHUTDOWN=0
#   RUN_ID=20260716a
#   FETCH_DATA=1                    # re-download Spec-Bench question.jsonl

set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"

# ==================== 配置 ====================
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B-Instruct}
TREE_MODEL_PATH=${TREE_MODEL_PATH:-/root/autodl-tmp/models/EAGLE3-LLaMA3.1-Instruct-8B}
BENCH=spec_bench
ARMS=${ARMS:-"baseline k1 k2"}
THETA=${THETA:-0.86}
BUDGET=${BUDGET:-8}
QUESTION_BEGIN=${QUESTION_BEGIN:-0}
QUESTION_END=${QUESTION_END:-480}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
MAX_CACHE_LEN=${MAX_CACHE_LEN:-4096}
RUN_ID=${RUN_ID:-fixed}
FETCH_DATA=${FETCH_DATA:-0}
SHUTDOWN=${SHUTDOWN:-1}
SPEC_BENCH_URL=${SPEC_BENCH_URL:-https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl}
FEISHU_WEBHOOK=${FEISHU_WEBHOOK:-}
# ==================== 配置结束 ====================

export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

QFILE="evaluation/data/${BENCH}/question.jsonl"
ANSWER_DIR="evaluation/data/${BENCH}/model_answer"
mkdir -p "$ANSWER_DIR"

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

ensure_spec_bench_data() {
  if [ -f "$QFILE" ] && [ "$FETCH_DATA" != "1" ]; then
    echo "using existing ${QFILE} ($(wc -l < "$QFILE" | tr -d ' ') lines)"
    return 0
  fi
  echo "fetching Spec-Bench question.jsonl → ${QFILE}"
  mkdir -p "evaluation/data/${BENCH}"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$QFILE" "$SPEC_BENCH_URL"
  else
    python3 - <<PY
import urllib.request
urllib.request.urlretrieve("${SPEC_BENCH_URL}", "${QFILE}")
PY
  fi
  echo "fetched $(wc -l < "$QFILE" | tr -d ' ') lines"
}

model_id_for() {
  local arm=$1
  echo "specbench__${arm}__${RUN_ID}"
}

answer_path_for() {
  local arm=$1
  echo "${ANSWER_DIR}/$(model_id_for "$arm").jsonl"
}

FAILED=""
RAN_ANY=0

run_one() {
  local arm=$1
  shift
  local model_id
  model_id=$(model_id_for "$arm")

  echo "==== $(date '+%F %T') START arm=${arm} id=${model_id} q=[${QUESTION_BEGIN},${QUESTION_END}) ===="
  if ! python3 evaluation/inference_samd.py \
    --model-id "$model_id" \
    --model-type llama3 --template llama3 \
    --model-path "$MODEL_PATH" --tree_model_path "$TREE_MODEL_PATH" \
    --bench-name "$BENCH" \
    --question-begin "$QUESTION_BEGIN" \
    --question-end "$QUESTION_END" \
    --tree_method eagle3 \
    --eagle3_total_token 60 \
    --samd_len_threshold 5 --samd_len_bias 5 \
    --max_cache_len "$MAX_CACHE_LEN" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    "$@"; then
    echo "==== FAILED ${model_id} ===="
    FAILED="${FAILED} ${model_id}"
  fi
  echo "==== $(date '+%F %T') DONE ${model_id} ===="
  RAN_ANY=1
}

# Per-domain + overall MAT/tok/s from answer jsonl (category field).
# Paper domains: multi_turn (8 MT-Bench cats) + 5 single-domain cats.
summarize_spec_bench() {
  local out_path=$1
  shift
  ANSWER_PATHS="$*" OUT_PATH="$out_path" python3 - <<'PY'
import json
import os
import sys
from collections import defaultdict

DOMAIN_OF = {
    "writing": "multi_turn",
    "roleplay": "multi_turn",
    "reasoning": "multi_turn",
    "math": "multi_turn",
    "coding": "multi_turn",
    "extraction": "multi_turn",
    "stem": "multi_turn",
    "humanities": "multi_turn",
    "translation": "translation",
    "summarization": "summarization",
    "qa": "qa",
    "math_reasoning": "math_reasoning",
    "rag": "rag",
}
DOMAIN_ORDER = [
    "multi_turn",
    "translation",
    "summarization",
    "qa",
    "math_reasoning",
    "rag",
    "overall",
]


def load_stats(path):
    # domain -> aggregates
    buckets = defaultdict(lambda: {"q": 0, "tokens": 0, "steps": 0, "wall": 0.0})
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            domain = DOMAIN_OF.get(row.get("category"), "other")
            choice = row["choices"][0]
            tokens = sum(choice["new_tokens"])
            steps = sum(choice["decoding_steps"])
            wall = sum(choice["wall_time"])
            for key in (domain, "overall"):
                buckets[key]["q"] += 1
                buckets[key]["tokens"] += tokens
                buckets[key]["steps"] += steps
                buckets[key]["wall"] += wall
    out = {}
    for key, value in buckets.items():
        if value["steps"] <= 0 or value["wall"] <= 0:
            continue
        out[key] = {
            "q": value["q"],
            "tokens": value["tokens"],
            "steps": value["steps"],
            "wall": value["wall"],
            "mat": value["tokens"] / value["steps"],
            "tok_s": value["tokens"] / value["wall"],
        }
    return out


paths = [path for path in os.environ["ANSWER_PATHS"].split() if path]
if not paths:
    print("no answer files", file=sys.stderr)
    sys.exit(1)

rows = []
for path in paths:
    label = os.path.splitext(os.path.basename(path))[0]
    if not os.path.isfile(path):
        print(f"missing {path}", file=sys.stderr)
        continue
    rows.append((label, load_stats(path)))

if not rows:
    sys.exit(1)

baseline = rows[0][1]
lines = []
header = (
    f"{'domain':16s} {'model':36s} {'q':>4s} {'MAT':>7s} {'tok/s':>8s} "
    f"{'MAT+%':>7s} {'speedup':>8s}"
)
lines.append(header)
lines.append("-" * len(header))

for domain in DOMAIN_ORDER:
    if domain not in baseline:
        continue
    base = baseline[domain]
    for label, stats in rows:
        if domain not in stats:
            continue
        current = stats[domain]
        mat_gain = (current["mat"] / base["mat"] - 1.0) * 100.0
        speedup = current["tok_s"] / base["tok_s"]
        lines.append(
            f"{domain:16s} {label:36s} {current['q']:>4d} {current['mat']:>7.4f} "
            f"{current['tok_s']:>8.2f} {mat_gain:>+7.2f} {speedup:>8.4f}"
        )
    lines.append("")

text = "\n".join(lines).rstrip() + "\n"
out_path = os.environ["OUT_PATH"]
with open(out_path, "w") as fout:
    fout.write(text)
print(text, end="")
PY
}

ensure_spec_bench_data
if [ ! -f "$QFILE" ]; then
  echo "ERROR: missing ${QFILE}"
  notify_feishu "Spec-Bench fixed batch: missing ${QFILE} on $(hostname)"
  do_shutdown
  exit 1
fi

echo "==== $(date '+%F %T') SPEC-BENCH FIXED ===="
echo "BENCH=${BENCH} questions=$(wc -l < "$QFILE" | tr -d ' ')"
echo "ARMS=${ARMS}"
echo "THETA=${THETA} BUDGET=${BUDGET} RUN_ID=${RUN_ID}"
echo "QUESTION_BEGIN=${QUESTION_BEGIN} QUESTION_END=${QUESTION_END}"

for ARM in $ARMS; do
  case "$ARM" in
    baseline)
      run_one baseline --fusion_mode none --tree_fusion none
      ;;
    k1)
      run_one k1 \
        --fusion_mode drafter_mars \
        --drafter_mars_repair graft \
        --drafter_mars_theta "$THETA" \
        --drafter_mars_graft_horizon "$BUDGET" \
        --drafter_mars_max_grafts 1
      ;;
    k2)
      run_one k2 \
        --fusion_mode drafter_mars \
        --drafter_mars_repair graft \
        --drafter_mars_theta "$THETA" \
        --drafter_mars_graft_horizon "$BUDGET" \
        --drafter_mars_max_grafts 2
      ;;
    *)
      echo "Unknown arm: ${ARM} (expected baseline|k1|k2)"
      FAILED="${FAILED} bad_arm:${ARM}"
      ;;
  esac
done

SUMMARY_FILES=()
for ARM in $ARMS; do
  AP=$(answer_path_for "$ARM")
  if [ -f "$AP" ]; then
    SUMMARY_FILES+=("$AP")
  fi
done

OVERALL_SUMMARY="${ANSWER_DIR}/specbench_${RUN_ID}_speed.txt"
DOMAIN_SUMMARY="${ANSWER_DIR}/specbench_${RUN_ID}_by_domain.txt"

echo "==== $(date '+%F %T') SUMMARY ===="
if [ "${#SUMMARY_FILES[@]}" -gt 0 ]; then
  python3 scripts/speed.py "${SUMMARY_FILES[@]}" | tee "$OVERALL_SUMMARY" || FAILED="${FAILED} overall_summary"
  summarize_spec_bench "$DOMAIN_SUMMARY" "${SUMMARY_FILES[@]}" || FAILED="${FAILED} domain_summary"
else
  FAILED="${FAILED} no_answers"
fi

notify_feishu "Spec-Bench fixed batch 完成 $(date '+%F %T')
host: $(hostname)
RUN_ID=${RUN_ID} THETA=${THETA} BUDGET=${BUDGET} ARMS=${ARMS}
FAILED:${FAILED:- none}

overall:
$(head -12 "$OVERALL_SUMMARY" 2>/dev/null || echo missing)

by domain:
$(head -40 "$DOMAIN_SUMMARY" 2>/dev/null || echo missing)

files:
  ${OVERALL_SUMMARY}
  ${DOMAIN_SUMMARY}"

if [ "$RAN_ANY" != "1" ]; then
  echo "No runs executed."
  do_shutdown
  exit 1
fi

do_shutdown
[ -z "$FAILED" ]
