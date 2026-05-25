#!/usr/bin/env bash
# Phase2 bench driver — runs on the HOST (not inside ablit).
# Spins up sglang on a fresh port serving the variant, runs tbench-17 + HA20,
# optionally MMLU-Pro + EQ (when HA20 >= threshold or FULL=1), then writes a
# one-line CSV row via append_result.py.
#
# Usage: run_bench.sh <exp_name> <variant_tag> <vlm_path> [port]
set -uo pipefail
EXP=${1:?usage: run_bench.sh <exp_name> <variant_tag> <vlm_path> [port]}
TAG=${2:?usage: run_bench.sh <exp_name> <variant_tag> <vlm_path> [port]}
VLM_PATH=${3:?usage: run_bench.sh <exp_name> <variant_tag> <vlm_path> [port]}
PORT=${4:-$((30050 + RANDOM % 50))}
RUN_DIR=$HOME/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64
NETWORK=gemma4-e4b-soyuz-agenttrove-qlora-r64_default
SGLANG_IMG=lmsysorg/sglang:nightly-dev-cu13-20260522-c9153da5
SGLANG_NAME=sglang-p2-${TAG}
HF_TOKEN=$(grep '^HF_TOKEN=' "$RUN_DIR/.env" | cut -d= -f2-)

CSV=$RUN_DIR/phase2_logs/all_variants.csv
OUT=$RUN_DIR/phase2_logs/${TAG}_bench_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$OUT" "$RUN_DIR/phase2_logs"
exec > >(tee -a "$OUT/bench.log") 2>&1
echo "[$(date -u +%H:%M:%SZ)] bench start exp=$EXP tag=$TAG vlm=$VLM_PATH port=$PORT"

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
if [ "$UP" != "1" ]; then
    echo "[err] sglang failed to come up"
    docker logs --tail=200 "$SGLANG_NAME" | tail -80
    exit 2
fi

# --- tbench-17 (parallel=3) ---------------------------------------------
TBENCH_LOG=$OUT/tbench17.log
TBENCH_OUT=$OUT/tbench17
mkdir -p "$TBENCH_OUT"
MODEL_TAG=$TAG BASE_URL="http://$SGLANG_NAME:$PORT/v1" \
    OUT_ROOT=$TBENCH_OUT \
    PARALLEL=${TBENCH_PARALLEL:-3} \
    NETWORK=$NETWORK \
    bash /tmp/phase2/common/tbench17_solvable_p2.sh > "$TBENCH_LOG" 2>&1 || true
TBENCH_LINE=$(grep -E "PASS [0-9]+ / [0-9]+" "$TBENCH_LOG" | tail -1)
TBENCH_PASS=$(echo "$TBENCH_LINE" | awk '{for(i=1;i<=NF;i++) if($i=="PASS") print $(i+1)}' | head -1)
echo "[tbench] $TBENCH_LINE"

# --- HA20 (parallel=2) --------------------------------------------------
HA20_LOG=$OUT/ha20.log
HA20_OUTDIR=$OUT/ha20
mkdir -p "$HA20_OUTDIR"
OUT=$HA20_OUTDIR HERMES_AGENT_20_BASE_URL="http://172.18.0.1:$PORT/v1" \
    HERMES_AGENT_20_MODEL=$TAG \
    HERMES_AGENT_20_LABEL=${TAG}-ha20 \
    PARALLEL=${HA20_PARALLEL:-2} \
    bash /tmp/phase2/common/ha20_parallel_p2.sh > "$HA20_LOG" 2>&1 || true
HA20_LINE=$(grep -E "TOTAL: [0-9]+/20" "$HA20_LOG" | tail -1)
HA20_PASS=$(echo "$HA20_LINE" | grep -oE 'TOTAL: [0-9]+' | awk '{print $2}')
echo "[ha20] $HA20_LINE"

MMLU_VAL=""
EQ_VAL=""
NEED_FULL=0
if [ "${FULL:-0}" = "1" ]; then NEED_FULL=1; fi
THRESHOLD=${HA20_THRESHOLD:-8}
if [ -n "${HA20_PASS:-}" ] && [ "$HA20_PASS" -ge "$THRESHOLD" ]; then NEED_FULL=1; fi

if [ "$NEED_FULL" = "1" ]; then
    echo "[full] running MMLU-Pro + EQ (HA20=$HA20_PASS, threshold=$THRESHOLD, FULL=${FULL:-0})"
    MMLU_LOG=$OUT_DIR/mmlu_pro.log
    MMLU_OUT=$OUT/mmlu_pro
    bash /tmp/phase2/common/mmlu_pro_runner.sh "$TAG" "http://127.0.0.1:$PORT/v1" > $OUT/mmlu_pro.log 2>&1 || true
    MMLU_VAL=$(grep -E "mmlu_pro\s*\|.*exact_match" $OUT/mmlu_pro.log | head -1 | grep -oE '[0-9]\.[0-9]+' | head -1)
    echo "[mmlu] $MMLU_VAL"

    bash /tmp/phase2/common/eqbench3_runner.sh "$TAG" "http://127.0.0.1:$PORT/v1" > $OUT/eqbench3.log 2>&1 || true
    EQ_VAL=$(grep -E 'Rubric Score' $OUT/eqbench3.log | grep -oE '[0-9]+\.[0-9]+' | tail -1)
    echo "[eq] $EQ_VAL"
fi

# Append CSV row (inside ablit because append_result.py is there too — but it's pure stdlib)
python3 /tmp/phase2/common/append_result.py \
    --csv "$CSV" --exp "$EXP" --variant "$TAG" \
    --tbench-pass "${TBENCH_PASS:-0}" --tbench-total 17 \
    --ha20-pass "${HA20_PASS:-0}" --ha20-total 20 \
    --mmlu-pro "${MMLU_VAL}" --eqbench "${EQ_VAL}" \
    --strength 0.5

echo "[done] tag=$TAG tbench=${TBENCH_PASS:-?}/17 ha20=${HA20_PASS:-?}/20 mmlu=${MMLU_VAL:-—} eq=${EQ_VAL:-—}"
echo "OUT=$OUT"
