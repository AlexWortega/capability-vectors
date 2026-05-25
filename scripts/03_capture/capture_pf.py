"""Capture last-token residual on SOYUZ-merged (text-only) for PASS-vs-FAIL contrast."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/workspace/ckpts_qwen35_soyuz/clawd_soyuz_merged"   # text-only Qwen3_5ForCausalLM
RUN_DIR = Path("/workspace/capvec_pf")

def load_prompts(path):
    return [json.loads(l)["prompt"] for l in open(path)]

@torch.no_grad()
def capture_one(model, tok, text, max_tokens=2048):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] > max_tokens:
        ids = ids[:, -max_tokens:]
    ids = ids.to(model.device)
    out = model(input_ids=ids, output_hidden_states=True, return_dict=True, use_cache=False)
    last = [h[0, -1, :].float().cpu().numpy() for h in out.hidden_states]
    return np.stack(last, axis=0)

def main():
    refuse_path = RUN_DIR / "contrast_refuse.jsonl"
    comply_path = RUN_DIR / "contrast_comply.jsonl"
    out_path = RUN_DIR / "activations" / "pf_acts.npz"
    out_path.parent.mkdir(exist_ok=True, parents=True)

    refuse_prompts = load_prompts(refuse_path)
    comply_prompts = load_prompts(comply_path)
    print(f"refuse(FAIL)={len(refuse_prompts)}  comply(PASS)={len(comply_prompts)}", flush=True)

    print(f"[load] {MODEL_PATH}", flush=True); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    nl = model.config.num_hidden_layers
    dm = model.config.hidden_size
    print(f"[load] class={type(model).__name__} n_layers={nl} d_model={dm} took={time.time()-t0:.1f}s", flush=True)

    print("[capture refuse/FAIL]", flush=True)
    refuse = []
    for i,p in enumerate(refuse_prompts):
        refuse.append(capture_one(model, tok, p))
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(refuse_prompts)}", flush=True)

    print("[capture comply/PASS]", flush=True)
    comply = []
    for i,p in enumerate(comply_prompts):
        comply.append(capture_one(model, tok, p))
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(comply_prompts)}", flush=True)

    refuse_arr = np.stack(refuse, axis=0)
    comply_arr = np.stack(comply, axis=0)
    print(f"refuse={refuse_arr.shape}  comply={comply_arr.shape}", flush=True)
    np.savez(out_path, refuse=refuse_arr, comply=comply_arr)
    print(f"saved -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
