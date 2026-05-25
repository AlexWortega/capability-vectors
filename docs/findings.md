# Findings

What worked, what surprised us, what broke.

## Tool-call parser is the bigger fix than the abliteration itself

Running [HermesAgent-20](https://github.com/stevibe/HermesAgent-20) on baseline soyuz **without** `--tool-call-parser hermes` → `1/20 pass, avg=17`. Hermes runtime sees `tool_calls=null` in every response because the model emits text-form `<tool_call>{...}</tool_call>` blocks instead of the OpenAI `tool_calls` array.

Same baseline **with** `--tool-call-parser hermes` → `4/20 pass, avg=61.9` — sglang parses the text block and synthesises a proper `tool_calls`, runtime executes them.

Then abliteration v2 on top → `8/20 pass`. **The 1→4 jump is parser glue. The 4→8 jump is the actual abliteration delta.**

## Clean-fail filter changes the contrast more than the direction algorithm does

Sequence:

- **v1**: 246 raw negatives + strength=1.0 → 0/17 tbench, 1/20 HA20 (catastrophic — direction includes verifier-noise tasks and clobbers everything at full strength)
- **v2**: 60 / 60 with Gemini-filtered `CLEAN_FAIL` only + strength=0.5 → 3/17 tbench, 8/20 HA20

Two changes between them, but the **clean filter is what shifts the direction away from noise**; strength is just a damping.

Of 246 reward=0 negatives, only 11 (4.5%) were flagged non-CLEAN by Gemini — yet keeping that 4.5% in the contrast was enough to ruin v1.

## AUC=1.0 in the probe is a red flag, not a green one

With `D=2560` activation dim and `N=120` (60+60) samples, **any** linear classifier hits AUC=1.0 on the training set. We saw this for both Logistic Regression (`v5_LR`, L=31) and Ridge reward-regression (`v5_REG`, L=10).

Downstream:

| Method | Train-set AUC | tbench-17 | HA20 |
|---|---:|---:|---:|
| MEAN diff | 0.928 | **3/17** | 8/20 |
| LR probe | 1.000 ⚠ | 2/17 | 6/20 |
| SVD top-1 | 0.777 | 1/17 | **10/20** |
| REG ridge | 1.000 ⚠ | 1/17 | **10/20** |

Best HA20 came from the method (SVD) with the **lowest** train AUC, and the worst tbench from the methods with perfect train AUC. The right test for a direction is the downstream bench, not the linear-separability AUC.

## MMLU-Pro collapses 58.72% → 2%

Both v2 (mean) and v5_REG (ridge) abliterated variants score ~2% on MMLU-Pro (5-shot, no-think, lm-eval). Baseline soyuz is 58.72%.

Why: the FAIL bucket contains many MMLU-Pi-agent fails (model not knowing the answer to a question). The mean-diff direction extracts a vector that points away from "model knows the fact" toward "model guesses / refuses / hallucinates". Orthogonalising weights against this vector deletes the "knowing" component.

Fix (queued): build an **agent-only contrast** that drops MMLU-Pi from both PASS and FAIL buckets. Test whether HA20 gain survives without the MMLU-Pro penalty.

## EQbench3 drops 64.35 → 51.10

Similar story, less severe. Empathy/reasoning subtly depend on the same residual axes the abliteration removes.

## Multi-layer per-layer ortho underperforms single-layer

Hypothesis was that per-layer directions would be tighter (each layer encodes different aspects of the fail mode). Result was the opposite: v3-multi (per-layer ortho with each layer's own direction, only layers in the 8-24 range with AUC > 0.65) scored 6/20 HA20 vs v2 single-layer 8/20.

Likely cause: per-layer directions are noisier than the global mean direction at a well-chosen mid-layer. Compounding ortho across 17 layers stacks the per-layer noise. The mlabonne single-layer recipe is more robust than its multi-layer extension.

## "Tradeoff" is the inevitable shape of the result, not a calibration issue

Across every variant we built (`v1`-`v5`), no setting dominates baseline on every bench:

- baseline wins MMLU-Pro and EQ
- abliterated variants win HA20
- tbench-17 modestly favours baseline (5/17 vs 1-3/17)

The "fail direction" computed from this trajectory pool is collinear with several capability axes simultaneously. Ortho along it inevitably trades them. Either the contrast must be made much narrower (one capability) or the intervention must be lighter (e.g. low-rank activation steering at inference instead of weight ortho).
