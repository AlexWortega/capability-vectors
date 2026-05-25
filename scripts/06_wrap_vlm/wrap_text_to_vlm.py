"""Wrap abliterated text-only into VLM multimodal so sglang can serve."""
import os, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from safetensors.torch import load_file

os.environ["HF_HOME"] = "/workspace/.hf"
BASE = "Qwen/Qwen3.5-4B"
SRC = "/workspace/abliterated_soyuz"
DST = "/workspace/abliterated_soyuz_vlm"

print(f"[load] {BASE}", flush=True)
m_vlm = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
sd_keys = set(m_vlm.state_dict().keys())

# Load abliterated text state_dict
text_sd = {}
for fn in sorted(os.listdir(SRC)):
    if fn.endswith(".safetensors"):
        text_sd.update(load_file(os.path.join(SRC, fn), device="cpu"))
print(f"text SD keys: {len(text_sd)}", flush=True)

# Rename model.X → model.language_model.X
renamed = {}
matched = unmatched = 0
for k, v in text_sd.items():
    new_k = k
    if k.startswith("model.") and not k.startswith("model.language_model."):
        cand = "model.language_model." + k[len("model."):]
        if cand in sd_keys: new_k = cand
        elif k in sd_keys: new_k = k
        else: unmatched += 1; continue
    elif k in sd_keys: new_k = k
    elif k.startswith("lm_head") and ("language_model.lm_head.weight" in sd_keys):
        new_k = "language_model." + k
    else:
        unmatched += 1; continue
    renamed[new_k] = v; matched += 1
print(f"matched={matched} unmatched={unmatched}", flush=True)

missing, unexpected = m_vlm.load_state_dict(renamed, strict=False)
print(f"load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

os.makedirs(DST, exist_ok=True)
print(f"[save] -> {DST}", flush=True)
m_vlm.save_pretrained(DST, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE, trust_remote_code=True).save_pretrained(DST)
try: AutoProcessor.from_pretrained(BASE, trust_remote_code=True).save_pretrained(DST)
except Exception as e: print(f"proc skip {e}")
print("done")
