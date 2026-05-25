# Results summary

All numbers from runs on eva02 (RTX A6000) using sglang nightly + custom benches. See `/scripts/07_bench/` for the runners.

## Bench protocols

| Bench | N tasks | Scoring | Where |
|---|---:|---|---|
| **tbench-2** ("solvable-17" subset) | 17 | docker verifier exit 0 = pass | [tbench-2](https://github.com/laude-institute/terminal-bench) — used the 17 tasks that **at least one** sibling Qwen3.5-4B variant had passed before (`ckpt600`, `clawd-100/-200/-rft/-rift`); excludes pure-noise hard tasks |
| **HermesAgent-20** | 20 | real Hermes runtime + artifact / memory / cron / browser verifiers | [HermesAgent-20](https://github.com/stevibe/HermesAgent-20) |
| **MMLU-Pro** | 12 032 | exact-match, 5-shot | lm-eval, `--apply_chat_template`, `enable_thinking=false` |
| **EQbench3** | 45 scenarios | Gemini-3-flash rubric judge, 0-100 | [eqbench3](https://github.com/EQ-bench/eqbench3) |

## All variants

| # | Variant | Method | strength | Bench setup notes | tbench-17 | HA20 | MMLU-Pro | EQ |
|---|---|---|---:|---|---:|---:|---:|---:|
| 0 | `soyuz` (baseline) | — | — | sglang base + LoRA via `--lora-paths`, hermes parser | **5/17** | 4/20 | **58.72%** | **64.35** |
| 0a | `soyuz` no-parser baseline | — | — | same but WITHOUT `--tool-call-parser hermes` | — | 1/20 | — | — |
| 1 | v1 ablit | MEAN, 246 raw negs | 1.0 | catastrophic | 0/17 | 1/20 | — | — |
| 2 | [v2 ablit](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-abliterated-v2) | MEAN, 60/60 Gemini-clean | 0.5 | L=16 AUC=0.928 | 3/17 | **8/20** | 2.09% | 51.10 |
| 3 | [v3-multi ablit](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-abliterated-v3-multi) | per-layer ortho L8-24 | 0.5 | same contrast as v2 | 2/17 | 6/20 | — | — |
| 4 | v4 ablit | MEAN, soyuz-self-only | 0.5 | L=12 AUC=0.808 | — | 6/20 | — | — |
| 5 | v5_LR | LR probe, soyuz-only | 0.5 | L=31 AUC=1.000 ⚠ | 2/17 | 6/20 | — | — |
| 6 | v5_SVD | SVD top-1, soyuz-only | 0.5 | L=27 AUC=0.777 | 1/17 | **10/20** | — | — |
| 7 | v5_REG | Ridge reward-reg, soyuz-only | 0.5 | L=10 AUC=1.000 ⚠ | 1/17 | **10/20** | 2.06% | — |

## Per-task HermesAgent-20 wins (passes only)

| variant | passes |
|---|---|
| baseline soyuz | HA-03 refuse, HA-06 background, HA-09 skill-create, HA-20 ambiguous |
| v2 ablit | HA-03, HA-06, **HA-08** browser, HA-09, **HA-11** patch-skill, **HA-15** cron-delivery, **HA-18** approval-gated, **HA-19** deployment-retry |
| v3-multi | HA-01 memory-replace, HA-02 memory-curate, HA-03, HA-06, HA-09, HA-11 |

v2 picked up **5 new agent-execution tasks** vs baseline. v3-multi picked up the two **memory-tool** tasks (HA-01, HA-02) that v2 missed. **Disjoint gains** suggest a possible ensemble win.

## Sibling models (not abliterated) for context

| Model | tbench-17 | HA20 | MMLU-Pro | EQ |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3.5-4B` (base, no LoRA) | 4/17 | — | 58.98% | — |
| vram.cloud `qwen3.6-27b` | **13/17** | — | — | **87.95** |
| vram.cloud `qwen3.6-35b-a3b` | 9/17 | — | — | 85.85 |
| `qwen35-4b-clawd-rift` (3-stage SFT → ClawGym → RIFT) | 4/17 | — | — | — |

The abliterated 4B variants close part of the gap to the 27/35B-class on HermesAgent-20 specifically while losing on knowledge.
