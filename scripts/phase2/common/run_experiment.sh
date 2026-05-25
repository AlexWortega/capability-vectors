#!/usr/bin/env bash
# Phase2 single-experiment driver.
# Usage: run_experiment.sh <variant_tag> <run_dir> [<strength>]
# Steps: capture -> compute_dir -> abliterate -> wrap -> bench
set -euo pipefail
VARIANT=${1:?usage: run_experiment.sh <variant_tag> <run_dir> [strength]}
RUN_DIR=${2:?usage: run_experiment.sh <variant_tag> <run_dir> [strength]}
STRENGTH=${3:-0.5}
PORT=${PORT:-$((30050 + RANDOM % 50))}
ABLIT_OUT=/workspace/abliterated_soyuz_${VARIANT}
VLM_OUT=/workspace/abliterated_soyuz_${VARIANT}_vlm

echo "[$(date -u +%H:%M:%SZ)] run_experiment tag=$VARIANT run_dir=$RUN_DIR strength=$STRENGTH port=$PORT"

# 1. capture
if [ ! -f "$RUN_DIR/activations/pf_acts.npz" ]; then
    echo "[capture] $RUN_DIR"
    python3 /tmp/phase2/capture_pf_param.py --run-dir "$RUN_DIR"
fi

# 2. compute direction
DIR_PT=$(ls -t $RUN_DIR/vectors/pf_dir_L*.pt 2>/dev/null | head -1 || true)
if [ -z "$DIR_PT" ]; then
    echo "[compute_dir] $RUN_DIR"
    python3 /tmp/phase2/compute_dir_mean_param.py --run-dir "$RUN_DIR"
    DIR_PT=$(ls -t $RUN_DIR/vectors/pf_dir_L*.pt | head -1)
fi
echo "[dir] $DIR_PT"

# 3. abliterate (CPU; uses ~16GB RAM)
if [ ! -d "$ABLIT_OUT" ] || [ -z "$(ls $ABLIT_OUT/*.safetensors 2>/dev/null)" ]; then
    echo "[abliterate] -> $ABLIT_OUT (strength=$STRENGTH)"
    python3 /tmp/phase2/abliterate_param.py \
        --dir-pt "$DIR_PT" --out "$ABLIT_OUT" --strength "$STRENGTH"
fi

# 4. wrap to multimodal
if [ ! -d "$VLM_OUT" ] || [ -z "$(ls $VLM_OUT/*.safetensors 2>/dev/null)" ]; then
    echo "[wrap_vlm] $ABLIT_OUT -> $VLM_OUT"
    SRC="$ABLIT_OUT" DST="$VLM_OUT" python3 /tmp/phase2/wrap_text_to_vlm_param.py
fi

# 5. bench (sglang serve + tbench17 + ha20)
bash /tmp/phase2/bench_variant.sh "$VARIANT" "$VLM_OUT" "$PORT"

echo "[$(date -u +%H:%M:%SZ)] run_experiment $VARIANT done"
