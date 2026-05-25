# exp5 — agent-only contrast

Hypothesis: the MMLU-Pro 58 → 2% collapse in v2/v5_REG is caused by `mmlu-pi-*`
trajectories living in the FAIL bucket. Drop them from both buckets and the
"failure direction" should stop being collinear with the knowledge axis.

Pipeline:

```
build_contrast_agent_only.py     # filter mmlu-pi from PASS+FAIL -> capvec_pf_v7_agentonly/
capture_pf.py                    # reuse phase1 script, point RUN_DIR -> v7
compute_pf_dir_mean.py           # reuse phase1 script, point RUN -> v7
abliterate_single_layer.py       # strength=0.5
wrap_text_to_vlm.py              # multimodal wrapper, sglang-servable
bench: tbench-17 + HA20 + MMLU-Pro + EQbench3
```

Success: HA20 >= 6/20 AND MMLU-Pro >= 40% (>= 60% of baseline 58.72%).
