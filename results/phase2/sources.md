# Phase 2 — contrast source distributions

All directions live in Qwen3.5-4B residual stream, **dim=2560**, mean-pooled at the last token of the last assistant message (v9 uses 5 positions). Layer chosen by max AUC on PASS-vs-FAIL classification.

## Source taxonomy

- **`mmlu-pi-*`** — MMLU-Pro routed through the **Pi-agent** (tool-call / function-call wrapper instead of MCQ)
  - `mmlu-pi-27b` — Qwen3.6-27b rollouts
  - `mmlu-pi-soyuz-keen` / `-flat` — soyuz at two sampling temperatures
- **`tbench-2-*`** — Terminal-Bench-2 trajectories: `-soyuz`, `-base`, `-27b`, `-35b`, `-rift`, `-dpo`, `-grpo`
- **`claw-eval-*`** — ClawGym eval rollouts (`-soyuz`, `-base30`)

## Vector inventory

| Variant | Method | Layer | AUC | dir.shape | HA20 | MMLU-Pro |
|---|---|---:|---:|---|---:|---:|
| v2 (mean) | mean(comply) − mean(refuse) | 16 | 0.928 | [2560] | **8/20** | 2.09% |
| v2 (pca) | top-1 SVD | 15 | — | [2560] | — | — |
| v4 soyuz-only | mean diff | 12 | 0.808 | [2560] | 6/20 | — |
| v5_LR | LR probe coef | 31 | 1.000 ⚠ | [2560] | 6/20 | — |
| v5_SVD | top-1 SVD | 27 | 0.777 | [2560] | **10/20** | **1.85%** |
| v5_REG | Ridge regr on reward | 10 | 1.000 ⚠ | [2560] | **10/20** | 2.06% |
| v6_hardpairs | mean diff (51 same-task pairs) | — | — | [2560] | — | — |
| v7_agentonly | mean diff, NO MMLU | 6 | — | [2560] | **9/20** | ⏳ |
| v9_multitok (mean) | mean diff, 5 positions | 11 | 0.778 | [2560] | 5/20 | — |
| v9_multitok (pca) | top-1 SVD, 5 positions | 15 | — | [2560] | — | — |

⚠ AUC=1.000 = overfitting (D=2560 ≫ N≈120). v5_LR collapsed on bench despite perfect train AUC.

## Per-experiment source distribution

### v2 — `/workspace/capvec_pf_v2/contrast_*.jsonl` (60/60 Gemini-clean)

```
comply (60): mmlu-pi-27b=25  mmlu-pi-soyuz-keen=17  mmlu-pi-soyuz-flat=3
              tbench=11  claw-eval-soyuz=4
              → MMLU-Pi = 45/60 = 75%

refuse (60): mmlu-pi-27b=12  mmlu-pi-soyuz-keen=11  mmlu-pi-soyuz-flat=9
              tbench=19  claw-eval=9
              → MMLU-Pi = 32/60 = 53%
```

→ MMLU signal dominates the direction → MMLU-Pro collapse to 2.09%.

### v6_hardpairs — `/workspace/capvec_pf_v6_hardpairs/contrast_*.jsonl` (51/51, same task pairs)

```
comply (51): mmlu-pi-27b=26  mmlu-pi-soyuz-keen=12 → MMLU-Pi=38/51=75%
              tbench=13
refuse (51): mmlu-pi-soyuz-keen=22  mmlu-pi-soyuz-flat=9  mmlu-pi-27b=7 → MMLU-Pi=38/51=75%
              tbench=13
```

→ Same MMLU bias as v2.

### v9_multitok — `/workspace/capvec_pf_v9_multitok/contrast_*.jsonl` (60/60)

```
comply (60): mmlu-pi-soyuz-keen=43 (72%)  mmlu-pi-flat=6 → MMLU-Pi=49/60=82%
              claw-eval=7  tbench=4
refuse (60): mmlu-pi-soyuz-keen=34  mmlu-pi-flat=13 → MMLU-Pi=47/60=78%
              tbench=7  claw-eval=6
```

→ Worst MMLU bias of the set → predicted failure confirmed (HA20=5/20).

### v7_agentonly — `/workspace/capvec_pf_v7_agentonly/contrast_*.jsonl` (42/60) — **hypothesis test**

```
comply (42): tbench-27b=13  tbench-35b=8  claw-eval-soyuz=8  tbench-soyuz=4
              tbench-base=3  tbench-grpo=2  tbench-dpo=2  tbench-rift=2
              → MMLU-Pi = 0/42 = 0% ✅

refuse (60): claw-eval-base30=13  tbench-rift=11  tbench-dpo=6  tbench-soyuz=6
              tbench-base=5  tbench-35b=5  claw-eval-soyuz=6  tbench-grpo=5  tbench-27b=3
              → MMLU-Pi = 0/60 = 0% ✅
```

→ Pure agent-trace signal. Direction surfaces at **L=6** (vs L=10-16 in MMLU-heavy variants — a different circuit). HA20=9/20 already locked, MMLU pending.

## Takeaway

Every winner up to v7 carried 50-80% MMLU-Pi in both buckets — the discovered direction was "MMLU-style reasoning" mixed with "refuse-to-act", not pure agent capability. Ablating it nuked MMLU-Pro to ~2%. v7_agentonly is the first MMLU-free contrast; if its MMLU stays near baseline (58.72%), the source-bias hypothesis is confirmed.
