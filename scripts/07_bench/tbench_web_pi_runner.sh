#!/usr/bin/env bash
set -uo pipefail
RUN_DIR="$HOME/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64"
OUT_ROOT="$RUN_DIR/tbench2_soyuz/$(date -u +%Y%m%d_%H%M%S)-web89"
NETWORK=$(docker inspect sglang-soyuz -f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{end}}')
RUNNER=/tmp/terminus_runner_web.py
PARALLEL=${PARALLEL:-3}
MODEL_TAG="soyuz"
BASE_URL="http://sglang-soyuz:30014/v1"

TASKS=($(docker images --format '{{.Repository}}' | grep '^agentbench/terminal-bench-2/' | sed 's|agentbench/terminal-bench-2/||' | sort -u))

mkdir -p "$OUT_ROOT"
exec > >(tee -a "$OUT_ROOT/eval.log") 2>&1
echo "[init] N=${#TASKS[@]} parallel=$PARALLEL model=$MODEL_TAG base=$BASE_URL  WEB-ENABLED"
echo "task,rc,reward,grade,turns,dur_s" > "$OUT_ROOT/$MODEL_TAG.summary.csv"

run_one() {
    local task="$1"
    local workdir="$OUT_ROOT/$task"
    mkdir -p "$workdir"
    local t0=$(date +%s)
    docker run --rm --network "$NETWORK" -t -m 4096m \
        -v "$workdir:/work" -v "$RUNNER:/runner.py:ro" \
        --entrypoint bash "agentbench/terminal-bench-2/${task}:latest" -c "
python3 /runner.py --base-url '$BASE_URL' --model '$MODEL_TAG' --instruction /instruction.md --work /work --cwd /app --max-turns 30 --max-tokens 4096 > /work/runner.stdout 2>&1
cd /app 2>/dev/null || cd /work
mkdir -p /logs/verifier
timeout 300 bash /tests/test.sh > /work/verifier.log 2>&1
cp /logs/verifier/reward.txt /work/ 2>/dev/null || true
cp /logs/verifier/ctrf.json /work/ 2>/dev/null || true
chmod -R a+rwX /work
" > "$workdir/container.stdout" 2> "$workdir/container.stderr"
    local rc=$?
    local dur=$(( $(date +%s) - t0 ))
    local reward=$(cat "$workdir/reward.txt" 2>/dev/null || echo NA)
    local grade=unknown
    [ -f "$workdir/ctrf.json" ] && grade=$(python3 -c "import json; d=json.load(open('$workdir/ctrf.json')); s=d.get('results',{}).get('summary',{}); print('pass' if s.get('failed',1)==0 and s.get('passed',0)>0 else 'fail')" 2>/dev/null || echo unknown)
    local turns=$(python3 -c "import json; d=json.load(open('$workdir/result.json')); print(d.get('turns','?'))" 2>/dev/null || echo ?)
    echo "$task,$rc,$reward,$grade,$turns,$dur" >> "$OUT_ROOT/$MODEL_TAG.summary.csv"
    printf "[%(%H:%M:%S)T] %-40s grade=%s turns=%s dur=%ss\n" -1 "$task" "$grade" "$turns" "$dur"
}
export -f run_one
export OUT_ROOT NETWORK RUNNER BASE_URL MODEL_TAG

printf "%s\n" "${TASKS[@]}" | xargs -I{} -P "$PARALLEL" bash -c 'run_one "$@"' _ {}

echo "==== SUMMARY ===="
cat "$OUT_ROOT/$MODEL_TAG.summary.csv"
PASSES=$(awk -F, 'NR>1 && $4=="pass"' "$OUT_ROOT/$MODEL_TAG.summary.csv" | wc -l)
TOTAL=$(awk -F, 'NR>1' "$OUT_ROOT/$MODEL_TAG.summary.csv" | wc -l)
echo "soyuz-web PASS $PASSES / $TOTAL"
