#!/usr/bin/env bash
# exp3 — sweep (alpha, layer) on the running steered server via /steering endpoint,
# then run HA20 (light) at each setting. tbench-17 only on the alpha that wins HA20.
#
# Server must be already running at $STEER_URL (e.g. http://localhost:30090).
set -uo pipefail
STEER_URL=${STEER_URL:-http://localhost:30090}
TAG=${TAG:-soyuz_steered}
DIR_PT=${DIR_PT:?DIR_PT (path to .pt) required}
OUT_ROOT=${OUT_ROOT:-/tmp/exp3_sweep_$(date -u +%Y%m%d_%H%M%S)}
mkdir -p $OUT_ROOT

ALPHAS=(${ALPHAS:--2 -1 -0.5 0 0.5 1 2})
LAYERS=(${LAYERS:-10 16 27})

echo "alpha,layer,ha20_pass,ha20_total,log" > $OUT_ROOT/sweep.csv

HOST_IP=${HOST_IP:-172.18.0.1}
PORT=${STEER_URL##*:}
PORT=${PORT%%/*}
HA20_BASE="http://${HOST_IP}:${PORT}/v1"

for L in "${LAYERS[@]}"; do
    for A in "${ALPHAS[@]}"; do
        echo "[$(date -u +%H:%M:%SZ)] sweep L=$L alpha=$A"
        curl -fsS -X POST "$STEER_URL/steering" \
            -H 'content-type: application/json' \
            -d "{\"alpha\": $A, \"layer\": $L, \"dir_pt\": \"$DIR_PT\"}"
        echo
        # warmup
        curl -fsS -X POST "$STEER_URL/v1/chat/completions" \
            -H 'content-type: application/json' \
            -d "{\"model\":\"$TAG\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}],\"max_tokens\":4}" >/dev/null

        SUBOUT=$OUT_ROOT/L${L}_A${A//\./p}
        mkdir -p $SUBOUT
        OUT=$SUBOUT HERMES_AGENT_20_BASE_URL=$HA20_BASE \
            HERMES_AGENT_20_MODEL=$TAG \
            HERMES_AGENT_20_LABEL=${TAG}_L${L}_A${A//\./p} \
            PARALLEL=${HA20_PARALLEL:-2} \
            bash /tmp/phase2/ha20_parallel_p2.sh > $SUBOUT/ha20.log 2>&1 || true
        PASSES=$(grep -oE 'TOTAL: [0-9]+/20' $SUBOUT/ha20.log | tail -1 | grep -oE '[0-9]+' | head -1)
        echo "$A,$L,${PASSES:-0},20,$SUBOUT/ha20.log" >> $OUT_ROOT/sweep.csv
        echo "  -> HA20=$PASSES/20"
    done
done
echo "==== sweep done ===="
cat $OUT_ROOT/sweep.csv
