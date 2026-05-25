#!/usr/bin/env bash
# exp2 — reuse pre-built v6 hard pairs (51 PASS/FAIL pairs sharing task_id).
# Capture -> compute mean direction -> abliterate strength=0.5 -> wrap -> bench.
set -euo pipefail

RUN_DIR=/workspace/capvec_pf_v6_hardpairs
VARIANT=v7_hardpairs
ABLIT_OUT=/workspace/abliterated_soyuz_${VARIANT}
VLM_OUT=/workspace/abliterated_soyuz_${VARIANT}_vlm
STRENGTH=${STRENGTH:-0.5}

cd /tmp
ls "$RUN_DIR/contrast_comply.jsonl" "$RUN_DIR/contrast_refuse.jsonl" >/dev/null

# 1) Capture activations (last-token, same as phase1) on the hard-pair contrast
python3 /tmp/phase2/exp2_capture_pf.py --run-dir "$RUN_DIR"

# 2) Compute MEAN direction
python3 /tmp/phase2/exp2_compute_dir.py --run-dir "$RUN_DIR"

# 3) Pick the saved direction
DIR_PT=$(ls -t $RUN_DIR/vectors/pf_dir_L*.pt | head -1)
echo "[exp2] using direction: $DIR_PT"

# 4) Abliterate
python3 /tmp/phase2/exp2_abliterate.py --dir-pt "$DIR_PT" --out "$ABLIT_OUT" --strength "$STRENGTH"

# 5) Wrap into multimodal arch
SRC="$ABLIT_OUT" DST="$VLM_OUT" python3 /tmp/phase2/exp2_wrap_vlm.py
echo "[exp2] ablit dir=$ABLIT_OUT vlm dir=$VLM_OUT (run bench separately)"
