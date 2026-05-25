#!/usr/bin/env bash
# Phase 2 bench wrapper for one abliterated variant.
# Brings up sglang on a fresh port, runs tbench-17 + HA20, optionally MMLU-Pro + EQ.
# Usage:
#   bench_variant.sh <variant_tag> <vlm_model_path> [port]
# env:
#   FULL=1                # also run MMLU-Pro + EQ regardless of HA20 score
#   HA20_THRESHOLD=8      # min HA20 passes to trigger MMLU/EQ
set -uo pipefail

TAG=${1:?usage: bench_variant.sh <tag> <vlm_model_path> [port]}
MODEL_PATH=${2:?usage: bench_variant.sh <tag> <vlm_model_path> [port]}
PORT=${3:-30050}
RUN_DIR=$HOME/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64
NETWORK=${NETWORK:-gemma4-e4b-soyuz-agenttrove-qlora-r64_default}
OUT_DIR=${OUT_DIR:-$RUN_DIR/phase2_bench/${TAG}_$(date -u +%Y%m%d_%H%M%S)}
SGLANG_IMG=lmsysorg/sglang:nightly-dev-cu13-20260522-c9153da5
SGLANG_NAME=sglang-${TAG}
HA20_THRESHOLD=${HA20_THRESHOLD:-8}

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/bench.log") 2>&1
echo "[$(date -u +%H:%M:%SZ)] bench start tag=$TAG path=$MODEL_PATH port=$PORT out=$OUT_DIR"

cleanup() {
    docker rm -f "$SGLANG_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup

docker run -d --name "$SGLANG_NAME" --gpus all --shm-size 8g \
    --network "$NETWORK" \
    -p "$PORT":"$PORT" \
    -v "$MODEL_PATH":/model:ro \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/hermes_qwen.jinja:/etc/hermes_qwen.jinja:ro \
    "$SGLANG_IMG" \
    python3 -m sglang.launch_server \
      --model-path /model --served-model-name "$TAG" \
      --host 0.0.0.0 --port "$PORT" --dtype bfloat16 \
      --mem-fraction-static 0.55 --trust-remote-code \
      --tool-call-parser hermes \
      --chat-template /etc/hermes_qwen.jinja

echo "[wait] sglang up..."
for i in {1..120}; do
    if curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[ok] sglang up after ${i}*5s"
        break
    fi
    sleep 5
done
if ! curl -fsS "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[err] sglang failed to come up"
    docker logs --tail=120 "$SGLANG_NAME" || true
    exit 2
fi

# --- tbench-17 ----------------------------------------------------------
TBENCH_LOG=$OUT_DIR/tbench17.log
MODEL_TAG=$TAG BASE_URL="http://$SGLANG_NAME:$PORT/v1" \
    OUT_ROOT="$OUT_DIR/tbench17" \
    PARALLEL=${TBENCH_PARALLEL:-3} \
    bash /tmp/phase2/tbench17_solvable_p2.sh > "$TBENCH_LOG" 2>&1 || true
TBENCH_LINE=$(grep -E "PASS [0-9]+ / [0-9]+" "$TBENCH_LOG" | tail -1)
echo "[tbench] $TBENCH_LINE"

# --- HA20 ---------------------------------------------------------------
HA20_LOG=$OUT_DIR/ha20.log
HOST_IP=$(getent hosts host.docker.internal 2>/dev/null | awk '{print $1}' || true)
if [ -z "$HOST_IP" ]; then HOST_IP=172.18.0.1; fi
HERMES_AGENT_20_BASE_URL="http://${HOST_IP}:$PORT/v1" \
    HERMES_AGENT_20_MODEL=$TAG \
    HERMES_AGENT_20_LABEL=${TAG}-ha20 \
    PARALLEL=${HA20_PARALLEL:-2} \
    bash /tmp/phase2/ha20_parallel_p2.sh > "$HA20_LOG" 2>&1 || true
HA20_LINE=$(grep -E "TOTAL: [0-9]+/20" "$HA20_LOG" | tail -1)
HA20_PASS=$(echo "$HA20_LINE" | grep -oE 'TOTAL: [0-9]+' | awk '{print $2}')
echo "[ha20] $HA20_LINE"

NEED_FULL=0
if [ "${FULL:-0}" = "1" ]; then NEED_FULL=1; fi
if [ -n "${HA20_PASS:-}" ] && [ "$HA20_PASS" -ge "$HA20_THRESHOLD" ]; then NEED_FULL=1; fi

if [ "$NEED_FULL" = "1" ]; then
    echo "[full] HA20=$HA20_PASS >= $HA20_THRESHOLD or FULL=1, running MMLU-Pro + EQ"
    MMLU_LOG=$OUT_DIR/mmlu_pro.log
    bash /tmp/phase2/mmlu_pro_runner.sh "$TAG" "http://localhost:$PORT/v1" > "$MMLU_LOG" 2>&1 || true
    MMLU_LINE=$(grep -Ei "leaderboard_mmlu_pro|mmlu_pro" "$MMLU_LOG" | tail -3)
    echo "[mmlu]"; echo "$MMLU_LINE"

    EQ_LOG=$OUT_DIR/eqbench3.log
    bash /tmp/phase2/eqbench3_runner.sh "$TAG" "http://localhost:$PORT/v1" > "$EQ_LOG" 2>&1 || true
    EQ_LINE=$(grep -Ei "eq score|final score|overall" "$EQ_LOG" | tail -3)
    echo "[eq]"; echo "$EQ_LINE"
fi

echo "[done] tag=$TAG ha20=$HA20_PASS"
echo "OUT=$OUT_DIR"
