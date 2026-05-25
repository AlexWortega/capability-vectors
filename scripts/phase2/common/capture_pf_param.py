"""Phase 2 last-token capture, parametrized on run_dir.

Same algorithm as scripts/03_capture/capture_pf.py but with --run-dir flag
so we can point at v6_hardpairs / v7_agentonly / v8_cfact / ...
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "/workspace/ckpts_qwen35_soyuz/clawd_soyuz_merged"


def load_prompts(path: Path):
    return [json.loads(l)["prompt"] for l in open(path)]


@torch.no_grad()
def capture_one(model, tok, text, max_tokens=2048):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] > max_tokens:
        ids = ids[:, -max_tokens:]
    ids = ids.to(model.device)
    out = model(input_ids=ids, output_hidden_states=True, return_dict=True, use_cache=False)
    return np.stack([h[0, -1, :].float().cpu().numpy() for h in out.hidden_states], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    run = Path(args.run_dir)
    refuse_prompts = load_prompts(run / "contrast_refuse.jsonl")
    comply_prompts = load_prompts(run / "contrast_comply.jsonl")
    out_path = run / "activations" / "pf_acts.npz"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"refuse(FAIL)={len(refuse_prompts)}  comply(PASS)={len(comply_prompts)}", flush=True)

    print(f"[load] {args.model_path}", flush=True); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] class={type(model).__name__} n_layers={model.config.num_hidden_layers} "
          f"d_model={model.config.hidden_size} took={time.time()-t0:.1f}s", flush=True)

    def run_set(name, prompts):
        arrs = []
        for i, p in enumerate(prompts):
            arrs.append(capture_one(model, tok, p, args.max_tokens))
            if (i + 1) % 10 == 0:
                print(f"  [{name}] {i+1}/{len(prompts)}", flush=True)
        return np.stack(arrs, axis=0)

    print("[capture refuse/FAIL]", flush=True)
    refuse = run_set("FAIL", refuse_prompts)
    print("[capture comply/PASS]", flush=True)
    comply = run_set("PASS", comply_prompts)
    print(f"refuse={refuse.shape}  comply={comply.shape}", flush=True)
    np.savez(out_path, refuse=refuse, comply=comply)
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
