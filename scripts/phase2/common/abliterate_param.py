"""Phase 2 wrapper around scripts/05_abliterate/abliterate_single_layer.py
that takes --dir-pt / --out / --strength and runs on CPU (re-uses the same
algorithm as phase 1)."""
from __future__ import annotations
import argparse, time, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "/workspace/ckpts_qwen35_soyuz/clawd_soyuz_merged"


def get_orth_columns(W, r):
    proj_mag = r @ W
    proj = r.unsqueeze(1) * proj_mag.unsqueeze(0)
    return W - proj


def get_orth_rows(W, r):
    proj_mag = W @ r
    proj = proj_mag.unsqueeze(1) * r.unsqueeze(0)
    return W - proj


def find_writers(model):
    writers = []
    for li, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            writers.append((f"L{li}.self_attn.o_proj", layer.self_attn.o_proj))
        elif hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
            writers.append((f"L{li}.linear_attn.out_proj", layer.linear_attn.out_proj))
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            writers.append((f"L{li}.mlp.down_proj", layer.mlp.down_proj))
    return writers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    args = ap.parse_args()

    blob = torch.load(args.dir_pt, weights_only=False, map_location="cpu")
    L = blob.get("layer")
    r = blob["dir"].float()
    print(f"direction L={L} AUC={blob.get('auc',0):.3f} margin={blob.get('margin',0):+.3f} method={blob.get('method','mean')}", flush=True)

    print(f"[load CPU] {args.model_path}", flush=True); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    print(f"[load] took={time.time()-t0:.1f}s", flush=True)

    r_dev = r.to(device=model.device, dtype=torch.float32)

    print("[ortho embed_tokens]", flush=True)
    E = model.model.embed_tokens.weight.data.float()
    new = get_orth_rows(E, r_dev)
    if args.strength != 1.0:
        new = E - args.strength * (E - new)
    model.model.embed_tokens.weight.data.copy_(new.to(torch.bfloat16))

    writers = find_writers(model)
    print(f"[ortho] {len(writers)} residual-writer projections", flush=True)
    with torch.no_grad():
        for name, mod in writers:
            W = mod.weight.data.float()
            new = get_orth_columns(W, r_dev)
            if args.strength != 1.0:
                new = W - args.strength * (W - new)
            mod.weight.data.copy_(new.to(torch.bfloat16))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[save] -> {out}", flush=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    meta = {"source": args.model_path, "direction_pt": args.dir_pt, "layer": L,
            "auc": float(blob.get("auc", 0)), "margin": float(blob.get("margin", 0)),
            "strength": args.strength,
            "method": blob.get("method", "weight-ortho FAIL-direction removal")}
    (out / "abliterate_meta.json").write_text(json.dumps(meta, indent=2))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
