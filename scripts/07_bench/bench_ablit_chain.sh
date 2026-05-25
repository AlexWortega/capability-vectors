#!/usr/bin/env bash
set -uo pipefail
TAG=soyuz_ablit_v2
NET_BASE=http://sglang-ablit2:30031/v1
OUT_ROOT=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/bench_ablit_v2
mkdir -p $OUT_ROOT
LOG=$OUT_ROOT/chain.log
exec > >(tee -a $LOG) 2>&1
echo "[$(date)] start"

# tbench-17
OUT_T=$OUT_ROOT/tbench; mkdir -p $OUT_T
RUNNER=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/terminus_runner.py
TASKS=(build-pmars cobol-modernization constraints-scheduling extract-elf fix-git fix-ocaml-gc git-leak-recovery hf-model-inference kv-store-grpc modernize-scientific-stack nginx-request-logging openssl-selfsigned-cert password-recovery portfolio-optimization pytorch-model-cli pytorch-model-recovery sqlite-with-gcov)
echo "task,rc,reward,grade,turns,dur_s" > $OUT_T/$TAG.summary.csv
run_one() {
    local task=$1; local wd=$OUT_T/$task; mkdir -p $wd; local t0=$(date +%s)
    docker run --rm --network gemma4-e4b-soyuz-agenttrove-qlora-r64_default -t -m 4096m \
        -v "$wd:/work" -v "$RUNNER:/runner.py:ro" --entrypoint bash "agentbench/terminal-bench-2/${task}:latest" -c "
python3 /runner.py --base-url '$NET_BASE' --model '$TAG' --instruction /instruction.md --work /work --cwd /app --max-turns 30 --max-tokens 4096 > /work/runner.stdout 2>&1
cd /app 2>/dev/null || cd /work; mkdir -p /logs/verifier
timeout 300 bash /tests/test.sh > /work/verifier.log 2>&1
cp /logs/verifier/reward.txt /work/ 2>/dev/null || true
cp /logs/verifier/ctrf.json /work/ 2>/dev/null || true
chmod -R a+rwX /work
" > $wd/container.stdout 2> $wd/container.stderr
    local rc=$?; local dur=$(( $(date +%s) - t0 ))
    local grade=unknown
    [ -f $wd/ctrf.json ] && grade=$(python3 -c "import json; d=json.load(open('$wd/ctrf.json')); s=d.get('results',{}).get('summary',{}); print('pass' if s.get('failed',1)==0 and s.get('passed',0)>0 else 'fail')" 2>/dev/null || echo unknown)
    echo "$task,$rc,$(cat $wd/reward.txt 2>/dev/null || echo NA),$grade,?,$dur" >> $OUT_T/$TAG.summary.csv
    printf "[tbench] %-35s grade=%s\n" "$task" "$grade"
}
export -f run_one; export OUT_T RUNNER NET_BASE TAG
echo "=== tbench-17 ==="
printf "%s\n" "${TASKS[@]}" | xargs -I{} -P 3 bash -c 'run_one "$@"' _ {}
P=$(awk -F, 'NR>1 && $4=="pass"' $OUT_T/$TAG.summary.csv | wc -l)
N=$(awk 'NR>1' $OUT_T/$TAG.summary.csv | wc -l)
echo "tbench: $P/$N pass"

# HA20 parallel 2 (to avoid sglang overload like last time)
echo "=== HA20 ==="
OUT_H=$OUT_ROOT/ha20; mkdir -p $OUT_H
cd /tmp/HermesAgent-20
export HERMES_AGENT_20_BASE_URL=http://172.18.0.1:30031/v1
export HERMES_AGENT_20_API_KEY=dummy
export HERMES_AGENT_20_MODEL=soyuz_ablit_v2
export HERMES_AGENT_20_PROVIDER=openai
export HERMES_AGENT_20_LABEL=soyuz_ablit_v2
export HERMES_AGENT_20_IMAGE=hermesagent20-dev
run_ha() {
    local id=$1
    node /tmp/HermesAgent-20/scripts/run-scenarios.mjs --scenario $id > $OUT_H/${id}.log 2>&1
    grep -E "PASS|FAIL" $OUT_H/${id}.log | tail -1
}
export -f run_ha; export OUT_H
printf "HA-%02d\n" {1..20} | xargs -I{} -P 2 bash -c 'run_ha "$@"' _ {}
PH=$(grep -h '\[PASS\]' $OUT_H/HA-*.log 2>/dev/null | wc -l)
echo "HA20: $PH/20 pass"
echo "[$(date)] DONE"
