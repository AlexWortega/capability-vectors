"""exp1 — capture residuals at multiple positions per trajectory.

Positions: 256, 768, 1280, 1792, last (truncate to 2048).
Output: pf_acts_multi.npz with keys
  refuse  : [N, n_positions, L+1, D]
  comply  : [N, n_positions, L+1, D]
  positions : [n_positions]    (token index actually used)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_POSITIONS = [256, 768, 1280, 1792]   # plus "last" appended
DEFAULT_MAX_TOKENS = 2048
MODEL_PATH = "/workspace/ckpts_qwen35_soyuz/clawd_soyuz_merged"


def load_prompts(path: Path):
    return [json.loads(l)["prompt"] for l in open(path)]


@torch.no_grad()
def capture_one(model, tok, text, positions, max_tokens):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] > max_tokens:
        ids = ids[:, -max_tokens:]
    ids = ids.to(model.device)
    seq_len = ids.shape[1]

    # Clip / dedupe positions; always include "last".
    used = []
    for p in positions:
        if p < seq_len:
            used.append(p)
    used.append(seq_len - 1)
    used = sorted(set(used))

    out = model(input_ids=ids, output_hidden_states=True, return_dict=True, use_cache=False)
    # out.hidden_states: tuple of (L+1) tensors [1, seq, D]
    layers = []
    for h in out.hidden_states:
        layers.append(h[0, used, :].float().cpu().numpy())   # [n_used, D]
    arr = np.stack(layers, axis=1)   # [n_used, L+1, D]

    # Pad to fixed length so we can stack across trajectories.
    n_pos = len(positions) + 1
    if arr.shape[0] < n_pos:
        pad = np.tile(arr[-1:], (n_pos - arr.shape[0], 1, 1))
        arr = np.concatenate([arr, pad], axis=0)
    return arr, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/workspace/capvec_pf_v2")
    ap.add_argument("--out", default=None,
                    help="Output npz path; default <run-dir>/activations/pf_acts_multi.npz")
    ap.add_argument("--positions", type=int, nargs="+", default=DEFAULT_POSITIONS)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--model-path", default=MODEL_PATH)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "activations" / "pf_acts_multi.npz"
    out_path.parent.mkdir(exist_ok=True, parents=True)

    refuse_prompts = load_prompts(run_dir / "contrast_refuse.jsonl")
    comply_prompts = load_prompts(run_dir / "contrast_comply.jsonl")
    print(f"refuse(FAIL)={len(refuse_prompts)}  comply(PASS)={len(comply_prompts)}", flush=True)
    print(f"positions={args.positions} + last  max_tokens={args.max_tokens}", flush=True)

    print(f"[load] {args.model_path}", flush=True); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load] class={type(model).__name__} n_layers={model.config.num_hidden_layers} "
          f"d_model={model.config.hidden_size} took={time.time()-t0:.1f}s", flush=True)

    def run(name, prompts):
        arrs, used_all = [], []
        for i, p in enumerate(prompts):
            arr, used = capture_one(model, tok, p, args.positions, args.max_tokens)
            arrs.append(arr); used_all.append(used)
            if (i + 1) % 10 == 0:
                print(f"  [{name}] {i+1}/{len(prompts)}", flush=True)
        return np.stack(arrs, axis=0), used_all

    print("[capture refuse/FAIL]", flush=True)
    refuse_arr, refuse_used = run("FAIL", refuse_prompts)
    print("[capture comply/PASS]", flush=True)
    comply_arr, comply_used = run("PASS", comply_prompts)

    print(f"refuse={refuse_arr.shape}  comply={comply_arr.shape}", flush=True)
    np.savez(out_path,
             refuse=refuse_arr, comply=comply_arr,
             positions=np.array(args.positions + [-1], dtype=np.int64))
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
