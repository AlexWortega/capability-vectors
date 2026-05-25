#!/usr/bin/env bash
# Master orchestrator: runs exp5, exp2, exp1, exp4, exp3 in sequence.
# Each invocation waits for prior to complete before starting next.
# Skips exp if its vlm dir already exists AND already in results CSV.
set -uo pipefail
NOTIFY=/home/alexw/.claude/skills/ml-intern/scripts/notify.sh
PROG=/tmp/phase2/progress.md
CSV=/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/phase2_logs/all_variants.csv
mkdir -p $(dirname $CSV)

# Wait for sglang-win to die.
echo "[$(date -u +%H:%M:%SZ)] waiting for sglang-win to free GPU..."
until ! docker ps --format "{{.Names}}" | grep -q "^sglang-win$"; do
    sleep 60
done
echo "[$(date -u +%H:%M:%SZ)] sglang-win GONE. GPU free. Starting phase2 sweep."
bash $NOTIFY code_ready "phase2: sglang-win freed, starting exp5_agent_only" || true

# --- exp5 agent-only (v7) ---
bash /tmp/phase2/runner/run_one_exp.sh exp5_agent_only /workspace/capvec_pf_v7_agentonly v7_agentonly 30050 \
    2>&1 | tee -a /tmp/phase2/runner/master.log

# --- exp2 hard pairs (v6) ---
bash /tmp/phase2/runner/run_one_exp.sh exp2_hard_pairs /workspace/capvec_pf_v6_hardpairs v6_hardpairs 30051 \
    2>&1 | tee -a /tmp/phase2/runner/master.log

# --- exp1 multi-token ---
bash /tmp/phase2/runner/run_one_exp_multitok.sh exp1_multi_token /workspace/capvec_pf_v2 v9_multitok 30052 \
    2>&1 | tee -a /tmp/phase2/runner/master.log

# --- exp4 counterfactual (v8) ---
bash /tmp/phase2/runner/run_one_exp.sh exp4_counterfactual /workspace/capvec_pf_v8_cfact v8_cfact 30053 \
    2>&1 | tee -a /tmp/phase2/runner/master.log

# --- exp3 activation steering (deferred) ---
echo "[$(date -u +%H:%M:%SZ)] exp3 steering: deferred — needs custom transformers server."
echo "$(date -u +%FT%TZ) exp3_steering deferred (custom server postponed; ran 4/5)" >> $PROG

bash $NOTIFY train_done "phase2 sweep complete (exp5/exp2/exp1/exp4; exp3 deferred). See $CSV." || true
echo "[$(date -u +%H:%M:%SZ)] phase2 ALL DONE"
