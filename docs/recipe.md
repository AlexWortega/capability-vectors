# Recipe

Step-by-step for reproducing the `v2` variant (`HermesAgent-20: 8/20`, our highest-tbench abliterated).

## Prerequisites

- 1× A6000 (or similar 40GB+ GPU)
- `Qwen/Qwen3.5-4B` cached in `~/.cache/huggingface` (multimodal `Qwen3_5ForConditionalGeneration`)
- An adapter+merge of `AlexWortega/qwen35-4b-soyuz` into the text-only `Qwen3_5ForCausalLM` flavor at `/workspace/ckpts_qwen35_soyuz/clawd_soyuz_merged` (≈8.5 GB). Built via:
  ```python
  from peft import PeftModel
  from transformers import AutoModelForCausalLM
  base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="cpu")
  base = PeftModel.from_pretrained(base, "AlexWortega/qwen35-4b-soyuz").merge_and_unload()
  base.save_pretrained("/workspace/ckpts_qwen35_soyuz/clawd_soyuz_merged")
  ```
- A pool of pass/fail trajectories aggregated from prior evals (`/workspace/rift_dataset_v2/train.jsonl`)
- `OPENROUTER_API_KEY` for Gemini-3-flash classification of negatives

## Step 1 — build contrast

```bash
python scripts/01_build_contrast/v2_clean_negs.py
# writes /workspace/capvec_pf_v2/contrast_{comply,refuse}.jsonl
#   PASS pool: 60 trajectories with reward=1.0 from any source
#   FAIL pool: 60 trajectories with reward=0.0 from any source — but FIRST classified clean
```

## Step 2 — (already done) clean the FAIL bucket

```bash
OPENROUTER_API_KEY=... python scripts/02_classify/classify_fails_gemini.py
# Runs google/gemini-3-flash-preview as judge on every reward==0 trajectory
# Labels: CLEAN_FAIL | VERIFIER_ISSUE | TASK_TOO_HARD | AMBIGUOUS
# Writes clean_fails.jsonl + ambiguous_fails.jsonl
# Typical: 235/246 = 95.5% are CLEAN_FAIL
```

`v2_clean_negs.py` then samples only `CLEAN_FAIL` rows for the contrast.

## Step 3 — capture activations

```bash
python scripts/03_capture/capture_pf.py
# Loads soyuz_merged on GPU 0, feeds each contrast text up to last 2048 tokens,
# stores residual at the LAST token at every hidden-state layer.
# Writes /workspace/capvec_pf_v2/activations/pf_acts.npz
#   refuse: [60, 33, 2560]  comply: [60, 33, 2560]
# Takes ~5 minutes.
```

## Step 4 — compute direction

```bash
python scripts/04_compute_direction/compute_pf_dir_mean.py
# For each layer L:
#   d = mean(refuse[:,L]) - mean(comply[:,L])
#   d /= ||d||
#   AUC = ranking ability of (X @ d) to separate PASS from FAIL
# Picks best L by argmax(|AUC - 0.5|)
# Writes /workspace/capvec_pf_v2/vectors/pf_dir_L<N>.pt
# v2 picks L=16, AUC=0.928
```

## Step 5 — orthogonalize weights

```bash
python scripts/05_abliterate/abliterate_single_layer.py \
    --dir-pt /workspace/capvec_pf_v2/vectors/pf_dir_L16.pt \
    --out /workspace/abliterated_soyuz_v2 \
    --strength 0.5
# Loads soyuz_merged on CPU (RAM ~16 GB working set),
# for each residual writer (embed_tokens rows, every layer's o_proj and down_proj cols):
#   W_orth = W - r @ (r.T @ W)
#   W_new  = W - strength * (W - W_orth)
# Saves the modified safetensors. Takes ~1 minute.
```

## Step 6 — wrap into multimodal arch for sglang

sglang does not have a registered entry class for the text-only `Qwen3_5ForCausalLM` — only for `Qwen3_5ForConditionalGeneration`. Workaround: take the official multimodal `Qwen/Qwen3.5-4B` base, swap its text decoder state_dict with our abliterated one (key remap `model.X` → `model.language_model.X`), keep its vision tower as-is.

```bash
python scripts/06_wrap_vlm/wrap_text_to_vlm.py
# Writes /workspace/abliterated_soyuz_v2_vlm (~9 GB)
```

## Step 7 — serve and bench

```bash
docker run -d --name sglang-ablit --gpus all --shm-size 8g \
    --network gemma4-e4b-soyuz-agenttrove-qlora-r64_default \
    -p 30030:30030 \
    -v /workspace/abliterated_soyuz_v2_vlm:/model:ro \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp/hermes_qwen.jinja:/etc/hermes_qwen.jinja:ro \
    lmsysorg/sglang:nightly-dev-cu13-20260522-c9153da5 \
    python3 -m sglang.launch_server \
      --model-path /model --served-model-name soyuz_ablit \
      --host 0.0.0.0 --port 30030 --dtype bfloat16 \
      --mem-fraction-static 0.55 --trust-remote-code \
      --tool-call-parser hermes \
      --chat-template /etc/hermes_qwen.jinja
```

Critical flag: `--tool-call-parser hermes` converts the model's text-emitted `<tool_call>{json}</tool_call>` into the OpenAI `tool_calls` field. Without it, HermesAgent-20 sees 0 tool calls and scores **1/20**. With it, baseline soyuz scores **4/20** and abliterated-v2 scores **8/20**.

Then point any of the bench scripts at port 30030:

```bash
bash scripts/07_bench/tbench17_solvable.sh soyuz_ablit
bash scripts/07_bench/ha20_parallel.sh
bash scripts/07_bench/bench_winners_mmlu_eq.sh
```
