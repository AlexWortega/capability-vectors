#!/usr/bin/env bash
# MMLU-Pro runner (no-think, concurrent=16) for one variant.
# Usage: mmlu_pro_runner.sh <model_tag> <base_url(http://host:port/v1)>
set -uo pipefail
TAG=${1:?usage: mmlu_pro_runner.sh <tag> <base_url>}
BASE=${2:?usage: mmlu_pro_runner.sh <tag> <base_url>}
OUT=${OUT:-/tmp/mmlu_pro_${TAG}_$(date +%s)}
mkdir -p $OUT
/tmp/claw-eval/.venv/bin/lm_eval --model local-chat-completions \
    --model_args "base_url=${BASE}/chat/completions,model=${TAG},tokenizer=Qwen/Qwen3.5-4B,num_concurrent=16,max_retries=5,timeout=60" \
    --tasks mmlu_pro --batch_size 1 --apply_chat_template \
    --output_path $OUT \
    --gen_kwargs '{"max_gen_toks":512,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
echo "==MMLU_DONE=="
grep -E "mmlu_pro|exact_match" $OUT/*.json 2>/dev/null | tail -20
