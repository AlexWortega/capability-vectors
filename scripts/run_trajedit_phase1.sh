#!/usr/bin/env bash
# TrajEdit Phase 1 — full pipeline runner for eva02 / A6000
# Run this from the capability-vectors repo root.
set -euo pipefail

MODEL_8B="${MODEL_8B:-Qwen/Qwen3-8B-Instruct}"
SGLANG_PORT="${SGLANG_PORT:-30040}"
WORKSPACE="${WORKSPACE:-/workspace/trajedit}"
HA20_SCENARIOS="${HA20_SCENARIOS:-/tmp/HermesAgent-20/scenarios}"

ROLLOUTS="$WORKSPACE/rollouts_qwen3_8b_ha20.jsonl"
LOCALIZED="$WORKSPACE/rollouts_localized.jsonl"
WITH_RECOVERY="$WORKSPACE/rollouts_with_recovery.jsonl"
ACTS_DIR="$WORKSPACE/activations"
VEC_DIR="$WORKSPACE/vectors"
EDIT_OUT="$WORKSPACE/trajedit_v1"
CTRL_OUT="$WORKSPACE/control_a_v1"

mkdir -p "$WORKSPACE"
BASE_URL="http://localhost:${SGLANG_PORT}/v1"

echo "========================================"
echo " TrajEdit Phase 1 — $(date -u)"
echo " model=$MODEL_8B  workspace=$WORKSPACE"
echo "========================================"

# ── Step 0: Serve Qwen3-8B ───────────────────────────────────────────────────
echo ""
echo "[step 0] Launch sglang server ($MODEL_8B → port $SGLANG_PORT)"
echo "  Run in a separate terminal:"
echo "    python -m sglang.launch_server \\"
echo "        --model-path $MODEL_8B \\"
echo "        --served-model-name qwen3_8b \\"
echo "        --port $SGLANG_PORT \\"
echo "        --tool-call-parser hermes \\"
echo "        --chat-template hermes_qwen.jinja \\"
echo "        --trust-remote-code"
echo ""
echo "  Waiting for server to become ready..."
until curl -s "$BASE_URL/models" > /dev/null 2>&1; do sleep 5; done
echo "  [ok] server ready"

# ── Step 1: Collect rollouts ──────────────────────────────────────────────────
echo ""
echo "[step 1] Collect rollouts → $ROLLOUTS"
python scripts/01_rollout/rollout_collector.py \
    --scenarios "$HA20_SCENARIOS" \
    --out "$ROLLOUTS" \
    --base-url "$BASE_URL" \
    --model qwen3_8b \
    --n-repeats 5 \
    --target-fail 200 \
    --target-pass 100

PASS_COUNT=$(python3 -c "import json; t=[ json.loads(l) for l in open('$ROLLOUTS')]; print(sum(1 for x in t if x['outcome']=='pass'))")
FAIL_COUNT=$(python3 -c "import json; t=[ json.loads(l) for l in open('$ROLLOUTS')]; print(sum(1 for x in t if x['outcome']=='fail'))")
echo "  [ok] pass=$PASS_COUNT fail=$FAIL_COUNT"

if [ "$FAIL_COUNT" -lt 50 ]; then
    echo "  [WARN] only $FAIL_COUNT FAIL traces — model may be too good on HA20."
    echo "  Consider adding more tasks or lowering max_turns."
fi

# ── Step 2a: First-error localization ────────────────────────────────────────
echo ""
echo "[step 2a] Prefix-flip localization → $LOCALIZED"
python scripts/02_localize/prefix_flip.py \
    --rollouts "$ROLLOUTS" \
    --out "$LOCALIZED" \
    --base-url "$BASE_URL" \
    --model qwen3_8b \
    --max-traces 200

# ── Step 2b: Recovery step extraction ────────────────────────────────────────
echo ""
echo "[step 2b] Recovery step extraction → $WITH_RECOVERY"
python scripts/02_localize/find_recovery.py \
    --rollouts "$LOCALIZED" \
    --out "$WITH_RECOVERY"

# ── Step 3: Activation capture (GPU intensive) ────────────────────────────────
echo ""
echo "[step 3] Capture residuals at localized steps → $ACTS_DIR"
echo "  (Shutting down sglang to free VRAM for capture...)"
pkill -f "sglang.launch_server.*${SGLANG_PORT}" 2>/dev/null || true
sleep 5

python scripts/03_capture/capture_trajedit.py \
    --rollouts "$WITH_RECOVERY" \
    --model-path "$MODEL_8B" \
    --out-dir "$ACTS_DIR"

# ── Step 4: Direction computation ─────────────────────────────────────────────
echo ""
echo "[step 4] Compute TrajEdit + outcome-only directions → $VEC_DIR"
python scripts/04_direction/compute_trajedit_dir.py \
    --acts-dir "$ACTS_DIR" \
    --out-dir "$VEC_DIR"

# Pick best layer from comparison JSON
BEST_TE_L=$(python3 -c "
import json
d = json.load(open('$VEC_DIR/layer_auc_comparison.json'))
print(d['best_trajedit']['layer'])
")
BEST_OC_L=$(python3 -c "
import json
d = json.load(open('$VEC_DIR/layer_auc_comparison.json'))
print(d['best_outcome_only']['layer'])
")
echo "  [ok] best TrajEdit layer=$BEST_TE_L  best outcome-only layer=$BEST_OC_L"

# ── Step 5a: TrajEdit-a (Steer2Edit rank-1) ──────────────────────────────────
echo ""
echo "[step 5a] TrajEdit-a: Steer2Edit rank-1 edit → $EDIT_OUT"
python scripts/05_edit/steer2edit.py \
    --dir-pt "$VEC_DIR/pf_dir_trajedit_L${BEST_TE_L}.pt" \
    --model-path "$MODEL_8B" \
    --out "$EDIT_OUT" \
    --strength 0.5 \
    --rho 0.5 \
    --alpha-en 0.5

# ── Step 5b: Control-A (outcome-only abliterate, current pipeline) ─────────────
echo ""
echo "[step 5b] Control-A: outcome-only mean-diff abliterate → $CTRL_OUT"
python scripts/05_abliterate/abliterate_single_layer.py \
    --dir-pt "$VEC_DIR/pf_dir_outcome_L${BEST_OC_L}.pt" \
    --out "$CTRL_OUT" \
    --strength 0.5

# ── Step 6: Wrap both for sglang ──────────────────────────────────────────────
echo ""
echo "[step 6] Wrap text→VLM for sglang serving"
python scripts/06_wrap_vlm/wrap_text_to_vlm.py --in "$EDIT_OUT"
python scripts/06_wrap_vlm/wrap_text_to_vlm.py --in "$CTRL_OUT"

# ── Step 7: Bench both variants ───────────────────────────────────────────────
echo ""
echo "[step 7] Bench — HA20 + MMLU-Pro"
echo ""
echo "  Run these in separate terminals (or adapt ha20_parallel.sh):"
echo ""
echo "  # TrajEdit-a"
echo "  python -m sglang.launch_server --model-path ${EDIT_OUT}_vlm \\"
echo "      --served-model-name trajedit_v1 --port 30041 \\"
echo "      --tool-call-parser hermes --chat-template hermes_qwen.jinja --trust-remote-code"
echo "  HERMES_AGENT_20_BASE_URL=http://localhost:30041/v1 \\"
echo "  HERMES_AGENT_20_MODEL=trajedit_v1 \\"
echo "      bash scripts/07_bench/ha20_parallel.sh"
echo ""
echo "  # Control-A"
echo "  python -m sglang.launch_server --model-path ${CTRL_OUT}_vlm \\"
echo "      --served-model-name control_a_v1 --port 30042 \\"
echo "      --tool-call-parser hermes --chat-template hermes_qwen.jinja --trust-remote-code"
echo "  HERMES_AGENT_20_BASE_URL=http://localhost:30042/v1 \\"
echo "  HERMES_AGENT_20_MODEL=control_a_v1 \\"
echo "      bash scripts/07_bench/ha20_parallel.sh"
echo ""
echo "========================================"
echo " Phase 1 complete.  Record results in:"
echo " ~/autoresearch-runs/trajedit/EXPERIMENTS.md"
echo "========================================"
