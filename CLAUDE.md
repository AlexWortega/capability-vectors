# capability-vectors

Weight-orthogonalization ("abliteration") experiments to steer a small agent LoRA model along **pass-vs-fail directions** computed from its own evaluation traces, without further training. Removes a residual-stream direction that correlates with task-failure → ortho weights → boosted agent benchmark behavior.

Base model: [`AlexWortega/qwen35-4b-soyuz`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz) (LoRA on `Qwen/Qwen3.5-4B` trained on Soyuz-sft).

## Pipeline (7 steps)

```
01_build_contrast  →  02_classify_negs  →  03_capture  →  04_compute_direction
                                                                ↓
                          07_bench  ←  06_wrap_vlm  ←  05_abliterate
```

1. **build_contrast** — pick PASS and FAIL trajectories from prior eval traces (claw-eval, tbench-2, MMLU-Pi-agent, HermesAgent-20). Render to chat-templated text blobs.
2. **classify** — Gemini-3-flash filters FAIL bucket to `CLEAN_FAIL` only (95.5% of 246 negatives passed this gate), drops `VERIFIER_ISSUE` / `AMBIGUOUS` / `TASK_TOO_HARD`.
3. **capture** — `AutoModelForCausalLM(soyuz_merged)`, feed each contrast text, store last-token residual at every layer → `[N, L+1, D]` arrays for PASS and FAIL.
4. **compute_direction** — per-layer compute one of:
   - **MEAN**: `μ_FAIL − μ_PASS`, normalized — original mlabonne recipe
   - **LR**: logistic-regression probe trained per layer, weight vector = direction (often AUC=1.0, overfit risk on D=2560, N=120)
   - **SVD**: top-1 singular vector of paired diff matrix
   - **REG**: ridge regression with continuous reward targets
   - **MULTI**: per-layer direction applied per-layer ortho (more capacity, less robust)
5. **abliterate** — orthogonalize the model's residual-writing projections (`embed_tokens` rows + every layer's `o_proj.weight` and `mlp.down_proj.weight` cols) against the chosen direction. Blendable via `strength α ∈ [0,1]`: `W ← W − α · (W − W_orth)`.
6. **wrap_vlm** — re-key text-only `Qwen3_5ForCausalLM` weights into the multimodal `Qwen3_5ForConditionalGeneration` shell (prepend `language_model.` to layer paths) so sglang can serve them — vision tower from base is preserved.
7. **bench** — sglang OpenAI-compat endpoint with `--tool-call-parser hermes` for agent benches (without it Hermes runtime sees 0 tool_calls).

## Variants and results

All variants are 4B `Qwen3.5-4B + Soyuz LoRA`, weight-orthogonalised against one (or many) direction(s). Bench setup: `tbench-2` 17-task solvable subset (parallel=3, max-turns=30), `HermesAgent-20` (parallel=2, native verifier), `MMLU-Pro` (5-shot, lm-eval, `enable_thinking=false`), `EQbench3` rubric (judge `google/gemini-3-flash-preview`).

| # | Variant | Contrast | Method | L | AUC | strength | tbench-17 | HA20 | MMLU-Pro | EQbench |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | `qwen35-4b-soyuz` | — | — | — | — | — | **5/17** | 4/20 | **58.72%** | **64.35** |
| v1 | `..-abliterated-v1` (not pushed) | 246 raw negs | MEAN | 11 | 0.886 | 1.0 | 0/17 | 1/20 | — | — |
| v2 | [`..-abliterated-v2`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-abliterated-v2) | 60 PASS / 60 Gemini-clean NEG (all sources) | MEAN | 16 | 0.928 | 0.5 | 3/17 | **8/20** | 2.09% | 51.10 |
| v3-multi | [`..-abliterated-v3-multi`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-abliterated-v3-multi) | same as v2 | per-layer ortho L8-24 | — | — | 0.5 | 2/17 | 6/20 | — | — |
| v4 | not pushed | 60 / 60 soyuz-self-traces only | MEAN | 12 | 0.808 | 0.5 | — | 6/20 | — | — |
| v5_LR | not pushed | soyuz-only | LR probe | 31 | 1.000⚠ | 0.5 | 2/17 | 6/20 | — | — |
| v5_SVD | not pushed | soyuz-only | SVD top-1 | 27 | 0.777 | 0.5 | 1/17 | **10/20** | — | — |
| v5_REG | not pushed | soyuz-only | reward-regression (ridge) | 10 | 1.000⚠ | 0.5 | 1/17 | **10/20** | 2.06% | — |

**Key findings**

- **HA20 gain is real**: best variants take HA20 from 4/20 → 10/20 (2.5×). Hermes runtime newly perceives tool_calls because abliteration removes the residual component that suppressed mid-trace tool emission.
- **MMLU-Pro collapses (58 → ~2)**: the "fail direction" overlaps with the **knowledge axis** because the contrast included MMLU-Pi-agent fails (model not knowing the answer). Orthogonalising it deletes knowledge. Solution: filter MMLU-Pi out of FAIL bucket; pass-only agent-style contrasts.
- **tbench-17 drops 1-3/17** (vs 5/17 baseline): same mechanism — clean shell scripting patterns partly overlap with the removed "give up" direction.
- **LR and REG hit AUC=1.0 on the contrast** but underperform in benches — classic overfit at D=2560 ≫ N=120. SVD with AUC=0.777 is best on HA20, suggesting AUC over-fit on the train set is anti-correlated with downstream gain.
- **strength=0.5** dominates `s=1.0` (which caused total collapse in v1) and `s>1.0` (untested, likely worse). Blending preserves baseline capabilities.

## Published HuggingFace models

| Repo | Description |
|---|---|
| [`AlexWortega/qwen35-4b-soyuz`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz) | Base LoRA on Qwen3.5-4B (Soyuz-sft) |
| [`AlexWortega/qwen35-4b-soyuz-merged`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-merged) | Merged bf16 of the above |
| [`AlexWortega/qwen35-4b-soyuz-abliterated-v2`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-abliterated-v2) | **8/20 HA20** — single-layer mean direction, clean negs |
| [`AlexWortega/qwen35-4b-soyuz-abliterated-v3-multi`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-abliterated-v3-multi) | 6/20 HA20 — per-layer multi-direction ortho |
| [`AlexWortega/qwen35-4b-clawd-rift`](https://huggingface.co/AlexWortega/qwen35-4b-clawd-rift) | (related ancestor; pre-Soyuz era) |
| [`AlexWortega/qwen35-4b-clawd-rift-merged`](https://huggingface.co/AlexWortega/qwen35-4b-clawd-rift-merged) | merged variant |
| [`AlexWortega/qwen35-4b-clawd-rift-gguf`](https://huggingface.co/AlexWortega/qwen35-4b-clawd-rift-gguf) | gguf quants |

## Layout

```
capability-vectors/
├── CLAUDE.md                       ← this file
├── README.md                       ← short pointer
├── scripts/
│   ├── 01_build_contrast/          ← pick PASS/FAIL trajectories, render to text
│   │   ├── v1_initial.py
│   │   ├── v2_clean_negs.py
│   │   ├── v4_soyuz_only.py
│   │   └── v6_hard_pairs.py
│   ├── 02_classify/                ← Gemini-3-flash filters FAIL bucket
│   │   └── classify_fails_gemini.py
│   ├── 03_capture/                 ← last-token residual capture per layer
│   │   └── capture_pf.py
│   ├── 04_compute_direction/       ← MEAN / LR / SVD / REG / MULTI
│   │   ├── compute_pf_dir_mean.py
│   │   └── dir_search_lr_svd_reg.py
│   ├── 05_abliterate/              ← weight orthogonalisation
│   │   ├── abliterate_single_layer.py
│   │   └── abliterate_multi_layer.py
│   ├── 06_wrap_vlm/                ← text → multimodal arch key remap
│   │   └── wrap_text_to_vlm.py
│   └── 07_bench/                   ← tbench-17, HA20, MMLU-Pro, EQbench3 runners
│       ├── bench_ablit_chain.sh
│       ├── bench_v5_methods.sh
│       ├── bench_winners_mmlu_eq.sh
│       ├── tbench17_solvable.sh
│       ├── tbench_web_pi_runner.sh
│       └── ha20_parallel.sh
├── docs/
│   ├── recipe.md                   ← step-by-step quickstart
│   └── findings.md                 ← what works, what breaks
└── results/
    └── summary.md                  ← all-variants result table
```

## Quickstart

Assumes a GPU box with `sglang` (nightly), `transformers >= 5.5`, `peft`, `scikit-learn`, and a fresh `OPENROUTER_API_KEY` for Gemini classification.

```bash
# 1. Build contrast (variant v2 — clean negs)
python scripts/01_build_contrast/v2_clean_negs.py

# 2. (optional) classify FAIL bucket with Gemini before step 3
OPENROUTER_API_KEY=... python scripts/02_classify/classify_fails_gemini.py

# 3. Capture activations on the soyuz-merged model
python scripts/03_capture/capture_pf.py

# 4. Compute direction (mean or LR/SVD/REG search)
python scripts/04_compute_direction/compute_pf_dir_mean.py
# or
python scripts/04_compute_direction/dir_search_lr_svd_reg.py

# 5. Abliterate (single-layer, strength 0.5)
python scripts/05_abliterate/abliterate_single_layer.py \
    --dir-pt /workspace/capvec_pf_v2/vectors/pf_dir_L16.pt \
    --out /workspace/abliterated_soyuz_v2 --strength 0.5

# 6. Wrap into multimodal arch so sglang serves it
python scripts/06_wrap_vlm/wrap_text_to_vlm.py

# 7. Serve and bench (sglang)
python -m sglang.launch_server --model-path /workspace/abliterated_soyuz_v2_vlm \
    --served-model-name soyuz_ablit --port 30030 --trust-remote-code \
    --tool-call-parser hermes --chat-template hermes_qwen.jinja
```

## Reusable contrast inventory (rift_dataset_v2)

The contrast pool comes from running soyuz (and several sibling models) on multiple agent + QA evals, then aggregating per-trajectory rewards:

- claw-eval `general` (175 tasks, Gemini-3-flash judge)
- tbench-2 17-task "solvable" subset across {soyuz, base, 27b, 35b, +rift, +dpo, +grpo}
- MMLU-Pro Pi-agent retry (100 hard MMLU questions, web-tool-enabled)
- HermesAgent-20 (20 tool-using scenarios)

`results/summary.md` has the full reward table.

## Outstanding directions (phase 2+)

- [ ] Multi-token capture (3-5 decision-point positions per trajectory, not just last-token)
- [ ] Per-task hard pairs (same prompt across runs that diverged pass/fail) — `v6_hard_pairs.py` built, 51 pairs
- [ ] Activation steering at inference time (hook into sglang to add `α·d` to layer-N residual) instead of weight ortho
- [ ] Counterfactual injection (force a wrong action mid-trace, capture residual at the divergence point)
- [ ] Agent-only contrast (drop MMLU-Pi from FAIL bucket — should rescue MMLU-Pro from collapse)
