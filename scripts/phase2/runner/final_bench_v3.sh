#!/usr/bin/env bash
# Final bench v3 — PRIORITY ORDER:
#  1. v7_agentonly MMLU+EQ (~4h)  -- critical hypothesis test
#  2. v8_cfact tbench+HA20 only (~1h)  -- need the HA20 score
#  3. (if time) v6_hardpairs MMLU+EQ (~4h)
set -uo pipefail
LOG=/tmp/phase2/runner/final_bench_v3.log
exec > >(tee -a $LOG) 2>&1
echo "[$(date -u +%H:%M:%SZ)] final_bench_v3 starting"

# Wait for sglang-soyuz-answers
while docker ps --format "{{.Names}}" | grep -q "^sglang-soyuz-answers$"; do sleep 60; done
echo "[$(date -u +%H:%M:%SZ)] sglang-soyuz-answers GONE."
while pgrep -f "lm_eval.*soyuz-answers" > /dev/null; do sleep 30; done
while pgrep -f "eqbench3.*soyuz-answers" > /dev/null; do sleep 30; done
echo "[$(date -u +%H:%M:%SZ)] GPU should be free. Sleeping 30s..."
sleep 30

cleanup() {
    for c in $(docker ps --format "{{.Names}}" | grep -E "^sglang-(p2|mmlu|final)-"); do
        echo "[cleanup] $c"
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
}
cleanup

# 1. v7_agentonly MMLU+EQ (PRIORITY)
echo "[$(date -u +%H:%M:%SZ)] === [P1] v7_agentonly MMLU+EQ start ==="
VLM=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/abliterated_soyuz_v7_agentonly_vlm
bash /tmp/phase2/runner/mmlu_eq_only.sh v7_agentonly "$VLM" 30060
echo "[$(date -u +%H:%M:%SZ)] === [P1] v7 done ==="
cleanup
sleep 30

# 2. v8_cfact HA20 only (skip MMLU/EQ via threshold)
echo "[$(date -u +%H:%M:%SZ)] === [P2] v8_cfact tbench+HA20 ==="
VLM=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/abliterated_soyuz_v8_cfact_vlm
HA20_THRESHOLD=21 bash /tmp/phase2/common/run_bench.sh exp4_counterfact v8_cfact "$VLM" 30062
echo "[$(date -u +%H:%M:%SZ)] === [P2] v8 done ==="
cleanup
sleep 30

# 3. v6_hardpairs MMLU+EQ
echo "[$(date -u +%H:%M:%SZ)] === [P3] v6_hardpairs MMLU+EQ ==="
VLM=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/abliterated_soyuz_v6_hardpairs_vlm
bash /tmp/phase2/runner/mmlu_eq_only.sh v6_hardpairs "$VLM" 30061
echo "[$(date -u +%H:%M:%SZ)] === [P3] v6 done ==="
cleanup

echo "[$(date -u +%H:%M:%SZ)] === ALL FINAL BENCHES DONE ==="
