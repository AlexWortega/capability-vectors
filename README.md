# capability-vectors

Weight-orthogonalization ("abliteration") experiments on [`Qwen3.5-4B + Soyuz LoRA`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz) using pass-vs-fail directions extracted from the model's own evaluation traces.

**See [CLAUDE.md](./CLAUDE.md) for the full recipe, results table, findings, and HuggingFace model links.**

## TL;DR

| Variant | tbench-17 | HermesAgent-20 | MMLU-Pro | EQ |
|---|---:|---:|---:|---:|
| baseline soyuz | 5/17 | 4/20 | 58.72% | 64.35 |
| [abliterated-v2](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-abliterated-v2) (single-L mean, s=0.5) | 3/17 | **8/20** | 2.09% ⚠ | 51.10 |
| [abliterated-v3-multi](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-abliterated-v3-multi) (per-layer ortho) | 2/17 | 6/20 | — | — |
| v5_SVD (top-1 SVD direction) | 1/17 | **10/20** | — | — |
| v5_REG (ridge reward-regression) | 1/17 | **10/20** | 2.06% ⚠ | — |

Method works for one capability axis (Hermes agent tool-use → 2.5×) but blasts general knowledge (MMLU-Pro → ~2%). Tradeoff comes from MMLU-Pi-agent fails being included in the contrast — work to fix is queued.

## Layout

`scripts/` is split per pipeline step (`01_build_contrast` → `07_bench`). All intermediate artifacts (contrasts, activations, directions, models) live under `/workspace/capvec_*` on the training box; only the source code is in git.
