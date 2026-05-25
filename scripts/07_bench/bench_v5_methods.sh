#!/usr/bin/env bash
set -uo pipefail
LOG=/tmp/bench_v5_chain.log
exec > >(tee -a $LOG) 2>&1

for NAME in LR SVD REG; do
    TAG=soyuz_ablit_v5_$NAME
    MODEL=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/abliterated_soyuz_v5_${NAME}_vlm
    OUT_ROOT=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/bench_ablit_v5_$NAME
    mkdir -p $OUT_ROOT
    PORT=$((30040 + RANDOM % 100))
    echo
    echo "============================================================"
    echo "[$(date)] BENCH $TAG on port $PORT"
    echo "============================================================"

    docker rm -f sglang-v5 2>&1 | tail
    HF_TOKEN=$(grep '^HF_TOKEN=' /home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/.env | cut -d= -f2-)
    docker run -d --name sglang-v5 --gpus all --network gemma4-e4b-soyuz-agenttrove-qlora-r64_default --shm-size 8g \
        -p $PORT:$PORT -e HF_TOKEN="$HF_TOKEN" -v $MODEL:/model:ro \
        -v /home/alexw/.cache/huggingface:/root/.cache/huggingface \
        -v /tmp/hermes_qwen.jinja:/etc/hermes_qwen.jinja:ro \
        lmsysorg/sglang:nightly-dev-cu13-20260522-c9153da5 \
        python3 -m sglang.launch_server --model-path /model --served-model-name $TAG \
        --host 0.0.0.0 --port $PORT --dtype bfloat16 --mem-fraction-static 0.65 \
        --trust-remote-code --tool-call-parser hermes --chat-template /etc/hermes_qwen.jinja 2>&1 | tail -1
    for i in $(seq 1 90); do
        curl -sS -m 2 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q $TAG && { echo "  sglang ready"; break; }
        docker ps --format '{{.Names}}' | grep -q sglang-v5 || { echo "  CRASHED"; docker logs sglang-v5 2>&1 | tail -8; continue 2; }
        sleep 5
    done

    # tbench-17
    OUT_T=$OUT_ROOT/tbench; mkdir -p $OUT_T
    RUNNER=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/terminus_runner.py
    NET_BASE="http://sglang-v5:$PORT/v1"
    TASKS=(build-pmars cobol-modernization constraints-scheduling extract-elf fix-git fix-ocaml-gc git-leak-recovery hf-model-inference kv-store-grpc modernize-scientific-stack nginx-request-logging openssl-selfsigned-cert password-recovery portfolio-optimization pytorch-model-cli pytorch-model-recovery sqlite-with-gcov)
    echo "task,rc,reward,grade" > $OUT_T/$TAG.summary.csv
    run_tbench() {
        local task=$1; local wd=$OUT_T/$task; mkdir -p $wd
        docker run --rm --network gemma4-e4b-soyuz-agenttrove-qlora-r64_default -t -m 4096m \
            -v "$wd:/work" -v "$RUNNER:/runner.py:ro" --entrypoint bash "agentbench/terminal-bench-2/${task}:latest" -c "
python3 /runner.py --base-url '$NET_BASE' --model '$TAG' --instruction /instruction.md --work /work --cwd /app --max-turns 30 --max-tokens 4096 > /work/runner.stdout 2>&1
cd /app 2>/dev/null || cd /work; mkdir -p /logs/verifier
timeout 300 bash /tests/test.sh > /work/verifier.log 2>&1
cp /logs/verifier/reward.txt /work/ 2>/dev/null || true
cp /logs/verifier/ctrf.json /work/ 2>/dev/null || true
chmod -R a+rwX /work
" > $wd/container.stdout 2> $wd/container.stderr
        local grade=unknown
        [ -f $wd/ctrf.json ] && grade=$(python3 -c "import json; d=json.load(open('$wd/ctrf.json')); s=d.get('results',{}).get('summary',{}); print('pass' if s.get('failed',1)==0 and s.get('passed',0)>0 else 'fail')" 2>/dev/null || echo unknown)
        echo "$task,$?,$(cat $wd/reward.txt 2>/dev/null || echo NA),$grade" >> $OUT_T/$TAG.summary.csv
        printf "[$NAME tbench] %-35s grade=%s\n" "$task" "$grade"
    }
    export -f run_tbench; export OUT_T RUNNER NET_BASE TAG NAME
    printf "%s\n" "${TASKS[@]}" | xargs -I{} -P 3 bash -c 'run_tbench "$@"' _ {}
    P=$(awk -F, 'NR>1 && $4=="pass"' $OUT_T/$TAG.summary.csv | wc -l)
    echo "[$NAME] tbench: $P / 17"

    # HA20 parallel 2
    OUT_H=$OUT_ROOT/ha20; mkdir -p $OUT_H
    cd /tmp/HermesAgent-20
    export HERMES_AGENT_20_BASE_URL=http://172.18.0.1:$PORT/v1
    export HERMES_AGENT_20_API_KEY=dummy
    export HERMES_AGENT_20_MODEL=$TAG
    export HERMES_AGENT_20_PROVIDER=openai
    export HERMES_AGENT_20_LABEL=$TAG
    export HERMES_AGENT_20_IMAGE=hermesagent20-dev
    run_ha() { node /tmp/HermesAgent-20/scripts/run-scenarios.mjs --scenario $1 > $OUT_H/$1.log 2>&1; grep -E 'PASS|FAIL' $OUT_H/$1.log | tail -1; }
    export -f run_ha; export OUT_H
    printf "HA-%02d\n" {1..20} | xargs -I{} -P 2 bash -c 'run_ha "$@"' _ {}
    PH=$(grep -h '\[PASS\]' $OUT_H/HA-*.log 2>/dev/null | wc -l)
    echo "[$NAME] HA20: $PH / 20"

    docker rm -f sglang-v5 2>&1 | tail
done
echo "[$(date)] V5 CHAIN DONE"
