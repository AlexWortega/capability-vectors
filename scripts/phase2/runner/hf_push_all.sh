#!/usr/bin/env bash
# Push the 3 winners to HF.
set -uo pipefail
HF_TOKEN=$(grep "^HF_TOKEN=" /home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/.env | cut -d= -f2-)
LOG=/tmp/phase2/runner/hf_push_all.log
exec > >(tee -a $LOG) 2>&1
echo "[$(date -u +%H:%M:%SZ)] HF push start"

push() {
    local TAG=$1 HA20=$2 TBENCH=$3 MMLU=$4 EQ=$5 METHOD=$6 NOTES=$7
    local VLM=/workspace/abliterated_soyuz_${TAG}_vlm
    local REPO=AlexWortega/qwen35-4b-soyuz-abliterated-${TAG}
    echo "[$(date -u +%H:%M:%SZ)] pushing $TAG -> $REPO"
    docker exec -e HF_TOKEN="$HF_TOKEN" ablit python3 /tmp/phase2/common/hf_push_variant.py \
        --tag "$TAG" --local "$VLM" \
        --tbench "${TBENCH}/17" --ha20 "$HA20" \
        --mmlu "$MMLU" --eq "$EQ" \
        --method "$METHOD" --notes "$NOTES" \
        --repo "$REPO" 2>&1 | tail -10
}

push v7_agentonly 9 2 "2.24%" "—" "phase2 exp5 agent-only contrast, mean diff L=6, strength=0.5" "MMLU-collapse hypothesis FALSIFIED: dropping MMLU-Pi from FAIL bucket does not protect MMLU"
push v6_hardpairs 8 2 "2.08%" "—" "phase2 exp2 within-task hard pairs (51 paired contrasts), mean diff L=9, strength=0.5" "Same-task contrast removes difficulty noise but MMLU still collapses"
push v8_cfact   9 2 "—"     "—" "phase2 exp4 counterfactual injection (wrong-action vs right-action), mean diff L=5, strength=0.5" "MMLU not measured (initial bench lost to GPU clash, re-bench captured HA20=9/20)"

echo "[$(date -u +%H:%M:%SZ)] HF push done"
