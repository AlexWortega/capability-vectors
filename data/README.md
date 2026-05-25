# `data/contrasts/` — the actual PASS / FAIL trajectories used to compute every direction

Each file is **JSONL**, one trajectory per line, schema:

```jsonc
{
  "prompt":  "<chat-templated text blob — full trace through the last assistant turn>",
  "source":  "tbench-2-soyuz | mmlu-pi-27b | claw-eval-soyuz | ...",
  "task_id": "sqlite-with-gcov | mmlu_pi_q3127 | ...",
  "reward":  1.0,        // comply only
  "reason":  "Gemini judge rationale for the failure"   // refuse only (v2/v7)
}
```

The activation-capture step (`scripts/03_capture/capture_pf.py`) feeds `prompt` verbatim to `AutoModelForCausalLM(soyuz_merged)` and stores the last-token residual at every layer. Direction-discovery (`scripts/04_compute_direction/*.py`) then operates on `comply` ⊕ `refuse` stacks.

## Files inventory

| File | rows | size | mean prompt chars | max chars |
|---|---:|---:|---:|---:|
| `contrast_v2_comply.jsonl`           | 60 | 267 KB | 4 090 | 12 000 |
| `contrast_v2_refuse.jsonl`           | 60 | 450 KB | 6 702 | 12 000 |
| `contrast_v4_soyuzonly_comply.jsonl` | 60 | 249 KB | 3 720 | 12 000 |
| `contrast_v4_soyuzonly_refuse.jsonl` | 60 | 337 KB | 5 284 | 12 000 |
| `contrast_v6_hardpairs_comply.jsonl` | 51 | 228 KB | 4 178 | 12 000 |
| `contrast_v6_hardpairs_refuse.jsonl` | 51 | 261 KB | 4 770 | 12 000 |
| `contrast_v7_agentonly_comply.jsonl` | 42 | 480 KB | **10 636** | 12 000 |
| `contrast_v7_agentonly_refuse.jsonl` | 60 | 694 KB | **10 059** | 12 000 |
| `contrast_v9_multitok_comply.jsonl`  | 60 | 249 KB | 3 720 | 12 000 |
| `contrast_v9_multitok_refuse.jsonl`  | 60 | 337 KB | 5 284 | 12 000 |

`v7_agentonly` prompts are **~2.5× longer** than every other variant — by design: dropping MMLU-Pi removes short single-turn QA traces, leaving long multi-turn agent trajectories.

`v9_multitok` reuses the v4 file content (same `source` distribution) but the capture script extracts residuals at 5 positions per trace instead of 1.

## Cross-tab source × variant_tag (raw counts)

| source ↓ \\ variant_tag → | v2 cmp | v2 ref | v4 cmp | v4 ref | v6 cmp | v6 ref | **v7 cmp** | **v7 ref** | v9 cmp | v9 ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mmlu-pi-soyuz-keen`  | 17 | 11 | 43 | 34 | 12 | 22 | **0** | **0** | 43 | 34 |
| `mmlu-pi-27b`         | 25 | 12 |  0 |  0 | 26 |  7 | **0** | **0** |  0 |  0 |
| `mmlu-pi-soyuz-flat`  |  3 |  9 |  6 | 13 |  0 |  9 | **0** | **0** |  6 | 13 |
| `claw-eval-soyuz`     |  4 |  5 |  7 |  6 |  0 |  0 |  8 |  6 |  7 |  6 |
| `claw-eval-base30`    |  0 |  4 |  0 |  0 |  0 |  0 |  0 | 13 |  0 |  0 |
| `tbench-2-soyuz`      |  2 |  4 |  4 |  7 |  1 |  1 |  4 |  6 |  4 |  7 |
| `tbench-2-27b`        |  7 |  1 |  0 |  0 |  5 |  0 | 13 |  3 |  0 |  0 |
| `tbench-2-35b`        |  1 |  2 |  0 |  0 |  4 |  0 |  8 |  5 |  0 |  0 |
| `tbench-2-base`       |  0 |  3 |  0 |  0 |  1 |  1 |  3 |  5 |  0 |  0 |
| `tbench-2-rift`       |  0 |  3 |  0 |  0 |  1 |  3 |  2 | 11 |  0 |  0 |
| `tbench-2-dpo`        |  1 |  5 |  0 |  0 |  0 |  4 |  2 |  6 |  0 |  0 |
| `tbench-2-grpo`       |  0 |  1 |  0 |  0 |  1 |  4 |  2 |  5 |  0 |  0 |
| **MMLU-Pi share %**   | **75%** | **53%** | **82%** | **78%** | **75%** | **75%** | **0%** | **0%** | **82%** | **78%** |
| **agent share %**     | 25% | 47% | 18% | 22% | 25% | 25% | **100%** | **100%** | 18% | 22% |

## What's actually in each variant — interpretation

### v2 — "the original Gemini-cleaned mix"
- Curated by Gemini-3-flash judge: kept 60 PASS + 60 FAIL marked `CLEAN_FAIL` (vs `VERIFIER_ISSUE` / `AMBIGUOUS` / `TASK_TOO_HARD`).
- 75% of PASS bucket = MMLU-Pi agent trajectories — model knew the answer.
- 53% of FAIL bucket = MMLU-Pi where the model didn't know.
- → Direction at L=16 (AUC=0.928) is **"model knows the MMLU answer"** mixed with "model emits a tool_call cleanly".
- Ablating it removes both: HA20 4 → 8 (good), MMLU 58.72 → 2.09 (catastrophic).

### v4 / v9 — "soyuz self-traces, mostly MMLU"
- Same file content; v9 differs only in capture (5 positions vs 1).
- 80%+ MMLU-Pi-soyuz in both buckets → same knowledge-axis bias as v2, even worse.
- v9 mean-aggregation: L=11 AUC=0.778, multi-token does **not** help.

### v6_hardpairs — "same task, different outcome"
- 51 task_ids appear in BOTH `comply` and `refuse` (verified: overlap=51/51).
- E.g. `mmlu_pi_q3127` PASSED by `mmlu-pi-27b` and FAILED by `mmlu-pi-soyuz-keen` (different sampling).
- Controls for task-difficulty confound, but **75% are still MMLU** → bias-axis problem remains.

### v7_agentonly — "drop MMLU, keep only agent" ★ hypothesis test
- **0% MMLU-Pi** in both buckets (verified by source column).
- PASS bucket: `tbench-2-27b`+`-35b`+`-soyuz` (long shell-scripting traces) + `claw-eval-soyuz`.
- FAIL bucket: 13× `claw-eval-base30` (Qwen3.5-4B base fails ClawGym) + 22× tbench across `rift`/`dpo`/`soyuz`/`grpo`/`base`/`27b`/`35b` (different variants failing different tasks).
- Direction surfaces at **L=6** — a much earlier layer than v2 (L=16) or v5_REG (L=10). Different circuit.
- Live result so far: HA20 = 9/20 (vs baseline 4, v5_SVD/REG winners 10). MMLU result pending — if ≥40%, the hypothesis "MMLU-Pi in FAIL bucket = knowledge-deletion" is confirmed and v7 is the first balanced winner.

## How a row got into a file

The pool is `/workspace/rift_dataset_v2/train.jsonl` (568 trajectories total across all evals). Each row was originally produced by:

1. **claw-eval-***: model rolled through the ClawGym (~175 tasks). Gemini-3-flash judged each completion 0-10; rows with judge ≥ 8 → `comply`, ≤ 3 → candidate for `refuse`.
2. **tbench-2-***: model executed Terminal-Bench-2 task in a docker sandbox; verifier `/tests/test.sh` exit code = pass/fail.
3. **mmlu-pi-***: model attempted a MMLU-Pro question wrapped as a Pi-agent dialog (`search → analyze → answer` tool chain). Final-answer exact-match vs ground truth.
4. **claw-eval-base30**: base `Qwen3.5-4B` (no LoRA) was forced through 30 hardest ClawGym tasks; all failed → contributes only to `refuse`.

Each variant's contrast-builder picks a subset by **source-filter + reward-rank + sample-cap**:
- `scripts/01_build_contrast/v2_clean_negs.py` — open-mixture, 60 + 60, post-Gemini filter
- `scripts/01_build_contrast/v4_soyuz_only.py` — only sources matching `soyuz` regex
- `scripts/01_build_contrast/v6_hard_pairs.py` — joins by `task_id`, keeps only IDs with both PASS and FAIL rollouts
- v7_agentonly — `v2_clean_negs.py` with `--exclude-mmlu` flag (drops every `mmlu-pi-*` source from both buckets)
- v9_multitok — symlinks v4 contrasts; only the capture differs

## Why this matters (the load-bearing finding)

The first 5 abliteration attempts (v1-v5) all fed the direction-discovery step a contrast that was **majority MMLU-Pi**. The discovered "fail direction" was therefore dominated by `model doesn't know the MMLU answer`, not by `model fails as agent`. Ablating that direction:
- ✅ Removed the residual that suppressed mid-trace tool emission (HA20 went up).
- ❌ Removed the residual that encodes factual recall (MMLU went from 58.72% → ~2%).

v7_agentonly is the minimum-change fix: **same algorithm, different data**. If it preserves MMLU and keeps HA20 ≥ 8, the bottleneck was never the algorithm — it was the contrast composition.
