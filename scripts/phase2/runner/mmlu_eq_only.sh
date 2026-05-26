#!/usr/bin/env bash
# MMLU+EQ only for one variant. Starts sglang, runs MMLU then EQ, tears down sglang.
# Usage: mmlu_eq_only.sh <tag> <vlm_path> <port>
set -uo pipefail
TAG=${1:?usage: mmlu_eq_only.sh tag vlm port}
VLM_PATH=${2:?}
PORT=${3:?}
NETWORK=${NETWORK:-gemma4-e4b-soyuz-agenttrove-qlora-r64_default}
SGLANG_IMG=lmsysorg/sglang:nightly-dev-cu13-20260522-c9153da5
SGLANG_NAME=sglang-mmlu-${TAG}
OUT=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/phase2_logs/${TAG}_mmlu_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
HF_TOKEN=$(grep "^HF_TOKEN=" /home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/.env | cut -d= -f2-)

echo "[$(date -u +%H:%M:%SZ)] MMLU+EQ only: tag=$TAG path=$VLM_PATH port=$PORT out=$OUT"

cleanup() { docker rm -f "$SGLANG_NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

docker run -d --name "$SGLANG_NAME" --gpus all --shm-size 8g \
    --network "$NETWORK" -p "$PORT":"$PORT" \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$VLM_PATH":/model:ro \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v /tmp/hermes_qwen.jinja:/etc/hermes_qwen.jinja:ro \
    "$SGLANG_IMG" \
    python3 -m sglang.launch_server \
      --model-path /model --served-model-name "$TAG" \
      --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
      --mem-fraction-static 0.55 --trust-remote-code \
      --tool-call-parser hermes \
      --chat-template /etc/hermes_qwen.jinja

echo "[wait] sglang up..."
UP=0
for i in $(seq 1 120); do
    if curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$TAG"; then
        UP=1; echo "[ok] up after ${i}*5s"; break
    fi
    sleep 5
done
if [ "$UP" != "1" ]; then echo "[err] sglang did not come up"; exit 2; fi

# MMLU-Pro
echo "[$(date -u +%H:%M:%SZ)] MMLU-Pro starting..."
bash /tmp/phase2/common/mmlu_pro_runner.sh "$TAG" "http://127.0.0.1:$PORT/v1" > "$OUT/mmlu_pro.log" 2>&1 || true
MMLU_VAL=$(grep -E "mmlu_pro\s*\|.*exact_match" "$OUT/mmlu_pro.log" | head -1 | grep -oE "0\.[0-9]+|[0-9]+\.[0-9]+" | head -1)
echo "[mmlu] $MMLU_VAL"

# EQbench3
echo "[$(date -u +%H:%M:%SZ)] EQbench3 starting..."
bash /tmp/phase2/common/eqbench3_runner.sh "$TAG" "http://127.0.0.1:$PORT/v1" > "$OUT/eqbench3.log" 2>&1 || true
EQ_VAL=$(grep -E "Rubric Score" "$OUT/eqbench3.log" | grep -oE "[0-9]+\.[0-9]+" | tail -1)
echo "[eq] $EQ_VAL"

echo "[$(date -u +%H:%M:%SZ)] DONE tag=$TAG mmlu=$MMLU_VAL eq=$EQ_VAL"
echo "OUT=$OUT"
