#!/usr/bin/env bash
set -euo pipefail

# Keep this path independent from any Lightning WORKDIR environment variable.
MAIN_REPO="/teamspace/studios/this_studio/SAM-Decoding"

if [ -f "${MAIN_REPO}/.env" ]; then
  # shellcheck disable=SC1090
  set -a && . "${MAIN_REPO}/.env" && set +a
fi

# Lightning shells may export WORKDIR=/home/zeus/content and project .env may
# set MAX_NEW_TOKENS for unrelated experiments. Keep this validation fixed.
WORKDIR="/teamspace/studios/this_studio/SAM-Decoding_validate_09c97b5_VuMjMm"
PY="/home/zeus/miniconda3/envs/cloudspace/bin/python"
MODEL_PATH="/teamspace/studios/this_studio/aicloud-data/Models/Meta-Llama-3.1-8B-Instruct"
TREE_MODEL_PATH="/teamspace/studios/this_studio/aicloud-data/Models/EAGLE3-LLaMA3.1-Instruct-8B"
BENCH_NAME="mt_bench"
MAX_NEW_TOKENS="1024"
MAX_CACHE_LEN="4096"
RUN_TAG="current_full_20260608"

cd "${WORKDIR}"
mkdir -p logs "evaluation/data/${BENCH_NAME}/model_answer"

notify() {
  local msg="$1"
  if [ -z "${FEISHU_WEBHOOK_URL:-}" ]; then
    echo "[notify] ${msg}"
    return 0
  fi
  MESSAGE="${msg}" "${PY}" - <<'PY' || true
import json
import os
import urllib.request

url = os.environ.get("FEISHU_WEBHOOK_URL")
msg = os.environ.get("MESSAGE", "")
if not url:
    print("NO_WEBHOOK")
else:
    data = json.dumps({"msg_type": "text", "content": {"text": msg}}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        print("FEISHU_STATUS", resp.status)
PY
}

write_machine_info() {
  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits || true
    else
      echo "nvidia-smi=missing"
    fi
    "${PY}" - <<'PY'
import torch
import transformers
print("torch={}".format(torch.__version__))
print("transformers={}".format(transformers.__version__))
print("cuda_available={}".format(torch.cuda.is_available()))
print("cuda_device_count={}".format(torch.cuda.device_count()))
PY
  } > "logs/${RUN_TAG}_machine.txt"
}

run_group() {
  local label="$1"
  local answer_file="$2"
  local log_file="$3"
  shift 3

  rm -f "${answer_file}" "${log_file}" "${log_file}.exit"
  notify "[dual-draft-fusion] START ${label} (${BENCH_NAME}) on $(hostname)"
  echo "START ${label} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "${log_file}.meta"

  if "${PY}" -m evaluation.inference_samd \
    --template llama3 \
    --model-type llama3 \
    --model-path "${MODEL_PATH}" \
    --model-id "${label}" \
    --bench-name "${BENCH_NAME}" \
    --num-choices 1 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype float16 \
    --samd_n_predicts 40 \
    --samd_len_threshold 5 \
    --samd_len_bias 5 \
    --tree_method eagle3 \
    --tree_model_path "${TREE_MODEL_PATH}" \
    --eagle3_total_token 60 \
    --eagle3_depth 7 \
    --eagle3_top_k 10 \
    --max_cache_len "${MAX_CACHE_LEN}" \
    --answer-file "${answer_file}" \
    "$@" > "${log_file}" 2>&1; then
    echo 0 > "${log_file}.exit"
    notify "[dual-draft-fusion] DONE ${label}"
  else
    local rc=$?
    echo "${rc}" > "${log_file}.exit"
    notify "[dual-draft-fusion] FAILED ${label} exit=${rc}; log=${WORKDIR}/${log_file}"
    return "${rc}"
  fi
}

write_summary() {
  "${PY}" - <<'PY' > "logs/current_full_mtbench_summary.txt"
import json
from pathlib import Path

files = {
    "pure_eagle3": Path("evaluation/data/mt_bench/model_answer/current_full_pure_eagle3.jsonl"),
    "sam_sequence_graft": Path("evaluation/data/mt_bench/model_answer/current_full_sam_sequence_graft.jsonl"),
    "naive_logprob": Path("evaluation/data/mt_bench/model_answer/current_full_naive_logprob.jsonl"),
}

def calc(path: Path):
    rows = turns = new_tokens = steps = 0
    wall_time = 0.0
    accepts = []
    qids = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows += 1
        row = json.loads(line)
        qids.append(row.get("question_id"))
        for choice in row.get("choices", []):
            new_tokens += sum(choice.get("new_tokens", []))
            wall_time += sum(choice.get("wall_time", []))
            steps += sum(choice.get("decoding_steps", []))
            accepts.extend(choice.get("accept_lengths", []))
            turns += len(choice.get("new_tokens", []))
    return {
        "rows": rows,
        "turns": turns,
        "new_tokens": new_tokens,
        "wall_time": wall_time,
        "tps": (new_tokens / wall_time) if wall_time else 0.0,
        "mat": (sum(accepts) / len(accepts)) if accepts else 0.0,
        "steps": steps,
        "qids": qids,
    }

stats = {name: calc(path) for name, path in files.items()}
base_tps = stats["pure_eagle3"]["tps"]
base_mat = stats["pure_eagle3"]["mat"]

print("# Current Full MT-Bench Summary")
print()
print("| Method | Rows | Turns | MAT | TPS | vs Eagle3 TPS | Steps |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for name, s in stats.items():
    ratio = s["tps"] / base_tps if base_tps else 0.0
    print("| {} | {} | {} | {:.3f} | {:.3f} | {:.3f}x | {} |".format(
        name, s["rows"], s["turns"], s["mat"], s["tps"], ratio, s["steps"]
    ))
print()
print("same_qids_all={}".format(
    stats["pure_eagle3"]["qids"] == stats["sam_sequence_graft"]["qids"] == stats["naive_logprob"]["qids"]
))
print("naive_mat_delta_vs_eagle3={:.3f}".format(stats["naive_logprob"]["mat"] - base_mat))
print("graft_mat_delta_vs_eagle3={:.3f}".format(stats["sam_sequence_graft"]["mat"] - base_mat))
print()
print(json.dumps(stats, indent=2, sort_keys=True))
PY
}

trap 'rc=$?; notify "[dual-draft-fusion] MT-Bench compare FAILED exit=${rc}; see ${WORKDIR}/logs/${RUN_TAG}.log"; exit ${rc}' ERR

write_machine_info
notify "[dual-draft-fusion] START full MT-Bench compare on $(hostname); tag=${RUN_TAG}"

run_group \
  "current_full_pure_eagle3" \
  "evaluation/data/${BENCH_NAME}/model_answer/current_full_pure_eagle3.jsonl" \
  "logs/current_full_pure_eagle3.log" \
  --fusion_mode none

run_group \
  "current_full_sam_sequence_graft" \
  "evaluation/data/${BENCH_NAME}/model_answer/current_full_sam_sequence_graft.jsonl" \
  "logs/current_full_sam_sequence_graft.log" \
  --fusion_mode none \
  --tree_fusion sam_sequence_graft

run_group \
  "current_full_naive_logprob" \
  "evaluation/data/${BENCH_NAME}/model_answer/current_full_naive_logprob.jsonl" \
  "logs/current_full_naive_logprob.log" \
  --fusion_mode naive \
  --fusion_max_draft_tokens 60 \
  --fusion_dedup_strategy max_score \
  --fusion_truncate_strategy score

write_summary
notify "[dual-draft-fusion] DONE full MT-Bench compare. Summary: ${WORKDIR}/logs/current_full_mtbench_summary.txt"
