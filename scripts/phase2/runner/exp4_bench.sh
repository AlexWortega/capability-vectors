#!/usr/bin/env bash
# Run exp4 v8_cfact bench (vlm already built). Skip MMLU/EQ via HA20_THRESHOLD=21.
set -uo pipefail
export HA20_THRESHOLD=21
LOG=/tmp/phase2/runner/exp4_bench.log
exec > >(tee -a $LOG) 2>&1
echo "[$(date -u +%H:%M:%SZ)] exp4 v8_cfact bench start"
VLM=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/abliterated_soyuz_v8_cfact_vlm
bash /tmp/phase2/common/run_bench.sh exp4_counterfact v8_cfact "$VLM" 30062
RC=$?
echo "[$(date -u +%H:%M:%SZ)] exp4 v8_cfact bench rc=$RC"
