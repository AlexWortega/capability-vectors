#!/usr/bin/env bash
# Phase2 single-experiment driver — runs INSIDE the ablit container.
# Steps: capture (GPU) -> compute_dir (CPU) -> abliterate (CPU) -> wrap (CPU).
# Bench/sglang launch must be done from the HOST afterwards (run_bench.sh).
# Usage: run_experiment.sh <variant_tag> <run_dir> [<strength>]
set -euo pipefail
VARIANT=${1:?usage: run_experiment.sh <variant_tag> <run_dir> [strength]}
RUN_DIR=${2:?usage: run_experiment.sh <variant_tag> <run_dir> [strength]}
STRENGTH=${3:-0.5}
ABLIT_OUT=/workspace/abliterated_soyuz_${VARIANT}
VLM_OUT=/workspace/abliterated_soyuz_${VARIANT}_vlm
DIR_PT=${DIR_PT:-}

echo "[$(date -u +%H:%M:%SZ)] run_experiment tag=$VARIANT run_dir=$RUN_DIR strength=$STRENGTH"

# 1. capture activations (only if not already present)
if [ ! -f "$RUN_DIR/activations/pf_acts.npz" ]; then
    echo "[capture] -> $RUN_DIR/activations/pf_acts.npz"
    python3 /tmp/phase2/common/capture_pf_param.py --run-dir "$RUN_DIR"
fi

# 2. compute direction (only if not already present)
if [ -z "$DIR_PT" ]; then
    DIR_PT=$(ls -t $RUN_DIR/vectors/pf_dir_L*.pt 2>/dev/null | head -1 || true)
fi
if [ -z "$DIR_PT" ] || [ ! -f "$DIR_PT" ]; then
    echo "[compute_dir] $RUN_DIR"
    python3 /tmp/phase2/common/compute_dir_mean_param.py --run-dir "$RUN_DIR"
    DIR_PT=$(ls -t $RUN_DIR/vectors/pf_dir_L*.pt | head -1)
fi
echo "[dir] $DIR_PT"

# 3. abliterate (CPU, ~16GB RAM working set)
if [ ! -f "$ABLIT_OUT/abliterate_meta.json" ]; then
    echo "[abliterate] -> $ABLIT_OUT (strength=$STRENGTH)"
    python3 /tmp/phase2/common/abliterate_param.py \
        --dir-pt "$DIR_PT" --out "$ABLIT_OUT" --strength "$STRENGTH"
fi

# 4. wrap to multimodal arch
if [ ! -d "$VLM_OUT" ] || [ -z "$(ls $VLM_OUT/*.safetensors 2>/dev/null)" ]; then
    echo "[wrap_vlm] $ABLIT_OUT -> $VLM_OUT"
    SRC="$ABLIT_OUT" DST="$VLM_OUT" python3 /tmp/phase2/common/wrap_text_to_vlm_param.py
fi

echo "[$(date -u +%H:%M:%SZ)] run_experiment $VARIANT pipeline done. Ready to bench."
echo "VLM_OUT=$VLM_OUT"
echo "DIR_PT=$DIR_PT"
