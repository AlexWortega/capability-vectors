#!/usr/bin/env bash
set -uo pipefail
LOG=/tmp/bench_winners.log
exec > >(tee -a $LOG) 2>&1

for VARIANT in "soyuz_ablit_v2:/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/abliterated_soyuz_v2_vlm" \
                "soyuz_ablit_v5_REG:/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/abliterated_soyuz_v5_REG_vlm" \
                "soyuz_ablit_v5_SVD:/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/abliterated_soyuz_v5_SVD_vlm"; do
    TAG=${VARIANT%%:*}; MODEL=${VARIANT##*:}
    PORT=$((30060 + RANDOM % 100))
    OUT=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/winner_bench/$TAG
    mkdir -p $OUT
    HOST_BASE="http://127.0.0.1:$PORT/v1"
    echo
    echo "============================================================"
    echo "[$(date)] FULL BENCH $TAG port=$PORT"

    docker rm -f sglang-win 2>&1 | tail
    HF_TOKEN=$(grep '^HF_TOKEN=' /home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/.env | cut -d= -f2-)
    docker run -d --name sglang-win --gpus all --network gemma4-e4b-soyuz-agenttrove-qlora-r64_default --shm-size 8g \
        -p $PORT:$PORT -e HF_TOKEN="$HF_TOKEN" -v $MODEL:/model:ro \
        -v /home/alexw/.cache/huggingface:/root/.cache/huggingface \
        -v /tmp/hermes_qwen.jinja:/etc/hermes_qwen.jinja:ro \
        lmsysorg/sglang:nightly-dev-cu13-20260522-c9153da5 \
        python3 -m sglang.launch_server --model-path /model --served-model-name $TAG \
        --host 0.0.0.0 --port $PORT --dtype bfloat16 --mem-fraction-static 0.65 \
        --trust-remote-code --tool-call-parser hermes --chat-template /etc/hermes_qwen.jinja 2>&1 | tail -1
    for i in $(seq 1 90); do
        curl -sS -m 2 http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q $TAG && break
        sleep 5
    done

    # MMLU-Pro (no-think, concurrent 16)
    echo "--- MMLU-Pro $TAG ---"
    /tmp/claw-eval/.venv/bin/lm_eval --model local-chat-completions \
        --model_args "base_url=$HOST_BASE/chat/completions,model=$TAG,tokenizer=Qwen/Qwen3.5-4B,num_concurrent=16,max_retries=5,timeout=60" \
        --tasks mmlu_pro --batch_size 1 --apply_chat_template \
        --output_path $OUT/mmlu_pro \
        --gen_kwargs '{"max_gen_toks":512,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
        > $OUT/mmlu.log 2>&1
    MM=$(grep mmlu_pro $OUT/mmlu.log | grep exact_match | tail -1)
    echo "MMLU: $MM"

    # EQbench3 (judge gemini-flash)
    echo "--- EQbench3 $TAG ---"
    cd /tmp/eqbench3
    OPENROUTER_API_KEY=$(grep '^OPENROUTER_API_KEY=' /home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/.env | cut -d= -f2-)
    cat > .env <<EOF
TEST_API_URL=$HOST_BASE/chat/completions
TEST_API_KEY=dummy
JUDGE_API_URL=https://openrouter.ai/api/v1/chat/completions
JUDGE_API_KEY=$OPENROUTER_API_KEY
MAX_RETRIES=4
RETRY_DELAY=5
REQUEST_TIMEOUT=300
LOG_VERBOSITY=INFO
EOF
    .venv/bin/python eqbench3.py --test-model "$TAG" --model-name "$TAG" --judge-model 'google/gemini-3-flash-preview' --threads 4 --no-elo --iterations 1 --runs-file eqbench3_runs_${TAG}.json > $OUT/eq.log 2>&1
    EQ=$(grep 'Rubric Score (0' $OUT/eq.log | tail -1)
    echo "EQ: $EQ"

    docker rm -f sglang-win 2>&1 | tail
    echo "[$(date)] DONE $TAG"
done
echo "[$(date)] ALL WINNERS DONE"
