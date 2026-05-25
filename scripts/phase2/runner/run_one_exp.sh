#!/usr/bin/env bash
# Phase2 single-experiment full pipeline: capture (GPU) -> compute -> ablit -> wrap -> bench
# Args: EXP_NAME RUN_DIR VARIANT_TAG [PORT]
# EXP_NAME = exp5_agent_only | exp2_hard_pairs | exp1_multi_token | exp4_counterfactual
# RUN_DIR  = /workspace/capvec_pf_v7_agentonly (etc)
# VARIANT_TAG = v7_agentonly (no soyuz_ prefix needed; common scripts add abliterated_soyuz_ prefix)
# Optional PORT.
set -uo pipefail
EXP=${1:?usage: run_one_exp.sh EXP RUN_DIR TAG [PORT]}
RUN_DIR=${2:?}
TAG=${3:?}
PORT=${4:-$((30050 + RANDOM % 50))}
STRENGTH=${STRENGTH:-0.5}
LOG_ROOT=/tmp/phase2/runner/logs/${EXP}_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
PROG=/tmp/phase2/progress.md
NOTIFY=/home/alexw/.claude/skills/ml-intern/scripts/notify.sh

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG_ROOT/main.log"; }
prog() { echo "$(date -u +%FT%TZ) $EXP $TAG  $*" >> "$PROG"; }

log "=== START $EXP tag=$TAG run_dir=$RUN_DIR port=$PORT ==="
prog "started"
bash $NOTIFY train_started "phase2 $EXP tag=$TAG (port $PORT) starting capture" || true

# Step 1: capture + compute_dir + abliterate + wrap (runs inside ablit container)
log "[step1] capture/dir/ablit/wrap in ablit container"
docker exec ablit bash /tmp/phase2/common/run_experiment.sh "$TAG" "$RUN_DIR" "$STRENGTH" \
    > "$LOG_ROOT/ablit_pipeline.log" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
    log "[FAIL] ablit pipeline rc=$RC"
    prog "FAIL ablit_pipeline rc=$RC"
    bash $NOTIFY error "phase2 $EXP ablit pipeline rc=$RC log=$LOG_ROOT/ablit_pipeline.log" || true
    exit 2
fi
VLM_PATH="/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/abliterated_soyuz_${TAG}_vlm"
ABLIT_PATH_CONTAINER="/workspace/abliterated_soyuz_${TAG}_vlm"
# Container has /workspace mapped to a host path; check the host symlink/bind:
# Conventional: ablit /workspace -> /home/alexw/runs/.../abliterated_soyuz_*_vlm? Check via docker inspect.
HOST_WORKSPACE=$(docker inspect ablit --format "{{range .Mounts}}{{if eq .Destination \"/workspace\"}}{{.Source}}{{end}}{{end}}")
log "host workspace mount: $HOST_WORKSPACE"
VLM_PATH="$HOST_WORKSPACE/abliterated_soyuz_${TAG}_vlm"
if [ ! -d "$VLM_PATH" ]; then
    log "[FAIL] vlm dir missing: $VLM_PATH"
    prog "FAIL vlm_missing $VLM_PATH"
    bash $NOTIFY error "phase2 $EXP missing vlm $VLM_PATH" || true
    exit 3
fi
log "[step1 done] VLM_PATH=$VLM_PATH"

# Step 2: bench on host
log "[step2] bench tbench-17 + HA20 (+ MMLU/EQ if HA20>=8) on host"
bash /tmp/phase2/common/run_bench.sh "$EXP" "$TAG" "$VLM_PATH" "$PORT" \
    > "$LOG_ROOT/bench.log" 2>&1
RC=$?
log "[bench done] rc=$RC"

# Extract result lines
TB_LINE=$(grep -E "^\[tbench\]" "$LOG_ROOT/bench.log" | tail -1)
HA_LINE=$(grep -E "^\[ha20\]" "$LOG_ROOT/bench.log" | tail -1)
MM_LINE=$(grep -E "^\[mmlu\]" "$LOG_ROOT/bench.log" | tail -1)
EQ_LINE=$(grep -E "^\[eq\]" "$LOG_ROOT/bench.log" | tail -1)
DONE_LINE=$(grep -E "^\[done\]" "$LOG_ROOT/bench.log" | tail -1)
log "$TB_LINE"; log "$HA_LINE"; log "$MM_LINE"; log "$EQ_LINE"; log "$DONE_LINE"

HA_PASS=$(echo "$DONE_LINE" | grep -oE "ha20=[0-9]+" | head -1 | cut -d= -f2)
TB_PASS=$(echo "$DONE_LINE" | grep -oE "tbench=[0-9]+" | head -1 | cut -d= -f2)
MM_VAL=$(echo "$DONE_LINE" | grep -oE "mmlu=[0-9.]+" | head -1 | cut -d= -f2)
EQ_VAL=$(echo "$DONE_LINE" | grep -oE "eq=[0-9.]+" | head -1 | cut -d= -f2)
prog "done tbench=${TB_PASS:-?}/17 ha20=${HA_PASS:-?}/20 mmlu=${MM_VAL:-—} eq=${EQ_VAL:-—}"

# Optional HF push if HA20 strong
if [ -n "${HA_PASS:-}" ] && [ "${HA_PASS}" -ge 8 ]; then
    log "[hf] HA20=$HA_PASS >= 8 -> pushing to HF"
    HF_TOKEN=$(grep "^HF_TOKEN=" /home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/.env | cut -d= -f2-)
    REPO="AlexWortega/qwen35-4b-soyuz-abliterated-${TAG}"
    HF_TOKEN="$HF_TOKEN" docker exec -e HF_TOKEN="$HF_TOKEN" ablit python3 /tmp/phase2/common/hf_push_variant.py \
        --tag "$TAG" --local "/workspace/abliterated_soyuz_${TAG}_vlm" \
        --tbench "${TB_PASS:-0}/17" --ha20 "${HA_PASS:-0}" \
        --mmlu "${MM_VAL:-—}" --eq "${EQ_VAL:-—}" \
        --method "phase2 $EXP, strength=$STRENGTH" \
        --notes "phase2 sweep" \
        --repo "$REPO" \
        > "$LOG_ROOT/hf_push.log" 2>&1 || true
    HF_URL=$(grep -E "https://huggingface.co/" "$LOG_ROOT/hf_push.log" | tail -1)
    log "[hf] $HF_URL"
    prog "HF $HF_URL"
fi

bash $NOTIFY train_done "phase2 $EXP $TAG: tbench=${TB_PASS:-?}/17 ha20=${HA_PASS:-?}/20 mmlu=${MM_VAL:-—} eq=${EQ_VAL:-—}" || true
log "=== END $EXP $TAG ==="
