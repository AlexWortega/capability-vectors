#!/usr/bin/env bash
set -uo pipefail
OUT=/tmp/ha20_par
mkdir -p $OUT
cd /tmp/HermesAgent-20
export HERMES_AGENT_20_BASE_URL=http://172.18.0.1:30025/v1
export HERMES_AGENT_20_API_KEY=dummy
export HERMES_AGENT_20_MODEL=soyuz
export HERMES_AGENT_20_PROVIDER=openai
export HERMES_AGENT_20_LABEL=soyuz-toolfix-par
export HERMES_AGENT_20_IMAGE=hermesagent20-dev

PARALLEL=${PARALLEL:-4}
SCENARIOS=$(printf "HA-%02d\n" {1..20})

run_one() {
    local id=$1
    node /tmp/HermesAgent-20/scripts/run-scenarios.mjs --scenario $id > $OUT/${id}.log 2>&1
    grep -E "PASS|FAIL" $OUT/${id}.log | tail -1 >> $OUT/_summary.log
    printf "[%(%H:%M:%S)T] %s done\n" -1 $id
}
export -f run_one
export OUT

echo "task,status,score" > $OUT/_csv.csv
printf "%s\n" $SCENARIOS | xargs -I{} -P $PARALLEL bash -c 'run_one "$@"' _ {}

# Parse all results
echo
echo "==== FINAL ===="
for id in $SCENARIOS; do
    line=$(grep -E "PASS|FAIL|PARTIAL" $OUT/${id}.log | tail -1)
    status=$(echo "$line" | grep -oE 'PASS|FAIL|PARTIAL' | head -1)
    score=$(echo "$line" | grep -oE 'score=[0-9]+' | head -1 | cut -d= -f2)
    echo "$id,$status,$score" >> $OUT/_csv.csv
    echo "  $id $status score=$score"
done
PASSES=$(awk -F, '$2=="PASS"' $OUT/_csv.csv | wc -l)
AVG=$(awk -F, 'NR>1 && $3!=""{s+=$3; n++} END{if(n) printf "%.1f", s/n}' $OUT/_csv.csv)
echo "TOTAL: $PASSES/20 pass  avg=$AVG"
