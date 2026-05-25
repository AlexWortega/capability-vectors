# exp4 — counterfactual injection

Replace the model's first tool-call assistant turn in 30 PASS trajectories
with a deliberately wrong action (`echo wrong`, `ls /nonexistent`,
`raise RuntimeError`). Feed both the original and the injected prefix
through soyuz, capture the residual at the last token of the decision turn.

Direction `d = mean(residual_injected - residual_original)`.

Uses standard `capture_pf.py` machinery (with `RUN_DIR` pointed at
`/workspace/capvec_pf_v8_cfact/`), then `compute_pf_dir_mean.py`, ablit,
wrap, bench.
