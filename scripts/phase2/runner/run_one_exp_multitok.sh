#!/usr/bin/env bash
# exp1 variant: uses multi-token capture + compute_dir, then standard abliterate/wrap/bench.
set -uo pipefail
EXP=${1:?usage}
RUN_DIR=${2:?}
TAG=${3:?}
PORT=${4:-30052}
STRENGTH=${STRENGTH:-0.5}
LOG_ROOT=/tmp/phase2/runner/logs/${EXP}_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$LOG_ROOT"
PROG=/tmp/phase2/progress.md
NOTIFY=/home/alexw/.claude/skills/ml-intern/scripts/notify.sh

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG_ROOT/main.log"; }
prog() { echo "$(date -u +%FT%TZ) $EXP $TAG  $*" >> "$PROG"; }

log "=== START $EXP tag=$TAG run_dir=$RUN_DIR port=$PORT ==="
prog "started multitok"
bash $NOTIFY train_started "phase2 $EXP tag=$TAG (multi-token capture)" || true

# Step 1a: multi-token capture
ACT_PATH="$RUN_DIR/activations/pf_acts_multi.npz"
if ! docker exec ablit test -f "$ACT_PATH"; then
    log "[capture multi-token] -> $ACT_PATH"
    docker exec ablit python3 /tmp/phase2/exp1_multi_token/capture_multi_token.py --run-dir "$RUN_DIR" \
        > "$LOG_ROOT/capture.log" 2>&1
    RC=$?
    if [ "$RC" -ne 0 ]; then
        log "[FAIL] capture rc=$RC"; prog "FAIL capture rc=$RC"
        bash $NOTIFY error "phase2 $EXP capture rc=$RC" || true; exit 2
    fi
fi
# Step 1b: compute multi-token direction
log "[compute_dir multi]"
docker exec ablit python3 /tmp/phase2/exp1_multi_token/compute_dir_multi.py --run-dir "$RUN_DIR" \
    > "$LOG_ROOT/compute_dir.log" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
    log "[FAIL] compute_dir rc=$RC"; prog "FAIL compute_dir rc=$RC"
    bash $NOTIFY error "phase2 $EXP compute_dir rc=$RC" || true; exit 3
fi
# pick best dir (the one saved without runner-up suffix order: first lines)
DIR_PT=$(docker exec ablit bash -c "ls -t $RUN_DIR/vectors/pf_dir_multi_*.pt 2>/dev/null | head -1")
log "selected dir: $DIR_PT"
if [ -z "$DIR_PT" ]; then
    log "[FAIL] no dir saved"; prog "FAIL no_dir"; exit 4
fi

# Step 1c: abliterate + wrap (override DIR_PT into common runner)
ABLIT_OUT=/workspace/abliterated_soyuz_${TAG}
VLM_OUT=/workspace/abliterated_soyuz_${TAG}_vlm
log "[ablit] -> $ABLIT_OUT (strength=$STRENGTH)"
docker exec ablit python3 /tmp/phase2/common/abliterate_param.py \
    --dir-pt "$DIR_PT" --out "$ABLIT_OUT" --strength "$STRENGTH" \
    > "$LOG_ROOT/abliterate.log" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
    log "[FAIL] abliterate rc=$RC"; prog "FAIL abliterate rc=$RC"
    bash $NOTIFY error "phase2 $EXP abliterate rc=$RC" || true; exit 5
fi
log "[wrap vlm]"
docker exec -e SRC="$ABLIT_OUT" -e DST="$VLM_OUT" ablit python3 /tmp/phase2/common/wrap_text_to_vlm_param.py \
    > "$LOG_ROOT/wrap.log" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
    log "[FAIL] wrap rc=$RC"; prog "FAIL wrap rc=$RC"
    bash $NOTIFY error "phase2 $EXP wrap rc=$RC" || true; exit 6
fi

HOST_WORKSPACE=$(docker inspect ablit --format "{{range .Mounts}}{{if eq .Destination \"/workspace\"}}{{.Source}}{{end}}{{end}}")
VLM_PATH="$HOST_WORKSPACE/abliterated_soyuz_${TAG}_vlm"
log "VLM_PATH=$VLM_PATH"

# Step 2: bench
log "[step2] bench"
bash /tmp/phase2/common/run_bench.sh "$EXP" "$TAG" "$VLM_PATH" "$PORT" \
    > "$LOG_ROOT/bench.log" 2>&1
RC=$?
DONE_LINE=$(grep -E "^\[done\]" "$LOG_ROOT/bench.log" | tail -1)
log "$DONE_LINE"
HA_PASS=$(echo "$DONE_LINE" | grep -oE "ha20=[0-9]+" | head -1 | cut -d= -f2)
TB_PASS=$(echo "$DONE_LINE" | grep -oE "tbench=[0-9]+" | head -1 | cut -d= -f2)
MM_VAL=$(echo "$DONE_LINE" | grep -oE "mmlu=[0-9.]+" | head -1 | cut -d= -f2)
EQ_VAL=$(echo "$DONE_LINE" | grep -oE "eq=[0-9.]+" | head -1 | cut -d= -f2)
prog "done tbench=${TB_PASS:-?}/17 ha20=${HA_PASS:-?}/20 mmlu=${MM_VAL:-—} eq=${EQ_VAL:-—}"

if [ -n "${HA_PASS:-}" ] && [ "${HA_PASS}" -ge 8 ]; then
    HF_TOKEN=$(grep "^HF_TOKEN=" /home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/.env | cut -d= -f2-)
    REPO="AlexWortega/qwen35-4b-soyuz-abliterated-${TAG}"
    HF_TOKEN="$HF_TOKEN" docker exec -e HF_TOKEN="$HF_TOKEN" ablit python3 /tmp/phase2/common/hf_push_variant.py \
        --tag "$TAG" --local "/workspace/abliterated_soyuz_${TAG}_vlm" \
        --tbench "${TB_PASS:-0}/17" --ha20 "${HA_PASS:-0}" \
        --mmlu "${MM_VAL:-—}" --eq "${EQ_VAL:-—}" \
        --method "phase2 exp1 multi-token, strength=$STRENGTH" \
        --notes "5-position residual aggregation" \
        --repo "$REPO" > "$LOG_ROOT/hf_push.log" 2>&1 || true
    prog "HF $(grep huggingface.co $LOG_ROOT/hf_push.log | tail -1)"
fi

bash $NOTIFY train_done "phase2 $EXP $TAG: tbench=${TB_PASS:-?}/17 ha20=${HA_PASS:-?}/20 mmlu=${MM_VAL:-—}" || true
log "=== END $EXP $TAG ==="
