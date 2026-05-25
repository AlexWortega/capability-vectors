#!/usr/bin/env bash
# EQbench3 runner (judge gemini-3-flash via openrouter).
# Usage: eqbench3_runner.sh <model_tag> <base_url(http://host:port/v1)>
set -uo pipefail
TAG=${1:?usage: eqbench3_runner.sh <tag> <base_url>}
BASE=${2:?usage: eqbench3_runner.sh <tag> <base_url>}
ENV_FILE=${ENV_FILE:-$HOME/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/.env}
OPENROUTER_API_KEY=$(grep '^OPENROUTER_API_KEY=' "$ENV_FILE" | cut -d= -f2-)
cd /tmp/eqbench3
cat > .env <<EOF
TEST_API_URL=${BASE}/chat/completions
TEST_API_KEY=dummy
JUDGE_API_URL=https://openrouter.ai/api/v1/chat/completions
JUDGE_API_KEY=${OPENROUTER_API_KEY}
MAX_RETRIES=4
RETRY_DELAY=5
REQUEST_TIMEOUT=300
LOG_VERBOSITY=INFO
EOF
.venv/bin/python eqbench3.py --test-model "$TAG" --model-name "$TAG" \
    --judge-model 'google/gemini-3-flash-preview' --threads 4 --no-elo --iterations 1 \
    --runs-file eqbench3_runs_${TAG}.json
