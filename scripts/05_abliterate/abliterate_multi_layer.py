"""Multi-layer abliteration: per-layer direction, ortho each layer's writers against its OWN direction.

Embedding: ortho against mean direction over chosen layer range.
"""
import argparse, time, json
from pathlib import Path
import torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/workspace/ckpts_qwen35_soyuz/clawd_soyuz_merged"

def get_orth_columns(W, r):
    proj_mag = r @ W
    proj = r.unsqueeze(1) * proj_mag.unsqueeze(0)
    return W - proj

def get_orth_rows(W, r):
    proj_mag = W @ r
    proj = proj_mag.unsqueeze(1) * r.unsqueeze(0)
    return W - proj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-npz", default="/workspace/capvec_pf_v2/activations/pf_acts.npz")
    ap.add_argument("--layer-range", default="8-24")  # which layers to use directions from
    ap.add_argument("--out", default="/workspace/abliterated_soyuz_v3_multi")
    ap.add_argument("--strength", type=float, default=0.5)
    args = ap.parse_args()

    z = np.load(args.acts_npz)
    refuse = z["refuse"]   # FAIL [N, L+1, D]
    comply = z["comply"]   # PASS
    n_layers = refuse.shape[1] - 1
    d_model = refuse.shape[2]
    lo, hi = map(int, args.layer_range.split("-"))
    print(f"n_layers={n_layers} d_model={d_model}, using layers {lo}-{hi}", flush=True)

    # Per-layer direction
    dirs = {}
    aucs = []
    for L in range(1, n_layers+1):
        mu_r = refuse[:, L].mean(0); mu_c = comply[:, L].mean(0)
        raw = mu_r - mu_c
        n = float(np.linalg.norm(raw))
        if n == 0: continue
        d = raw / n
        # AUC
        s_r = refuse[:, L] @ d; s_c = comply[:, L] @ d
        wins = sum(1 if p > nn else 0.5 if p == nn else 0 for p in s_r for nn in s_c)
        auc = wins / (len(s_r)*len(s_c))
        dirs[L] = (d, auc, n)
        aucs.append((L, auc))

    # Mean direction across selected range for embedding
    selected_dirs = []
    for L in range(lo, hi+1):
        if L in dirs and abs(dirs[L][1] - 0.5) > 0.2:
            selected_dirs.append(dirs[L][0])
    mean_d = np.mean(selected_dirs, axis=0)
    mean_d = mean_d / np.linalg.norm(mean_d)
    print(f"selected {len(selected_dirs)} layers for embed direction", flush=True)

    print(f"[load] {MODEL_PATH}", flush=True); t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    # Embedding: ortho against mean direction
    r_emb = torch.from_numpy(mean_d.astype(np.float32)).to(model.device)
    print(f"[ortho embed (mean dir)]", flush=True)
    E = model.model.embed_tokens.weight.data.float()
    new = get_orth_rows(E, r_emb)
    if args.strength != 1.0:
        new = E - args.strength * (E - new)
    model.model.embed_tokens.weight.data.copy_(new.to(torch.bfloat16))

    # Per-layer writers ortho against own layer's direction
    print(f"[ortho per-layer writers]", flush=True)
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            # Use this LAYER's direction (li+1 in dirs since dirs indexes by residual-after-layer)
            L_idx = li + 1
            if L_idx not in dirs:
                continue
            d_l, auc, _ = dirs[L_idx]
            if abs(auc - 0.5) < 0.15:  # skip weak directions
                continue
            r_dev = torch.from_numpy(d_l.astype(np.float32)).to(model.device)
            writers = []
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                writers.append((f"L{li}.self_attn.o_proj", layer.self_attn.o_proj))
            elif hasattr(layer, "linear_attn") and hasattr(layer.linear_attn, "out_proj"):
                writers.append((f"L{li}.linear_attn.out_proj", layer.linear_attn.out_proj))
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                writers.append((f"L{li}.mlp.down_proj", layer.mlp.down_proj))
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
    meta = {"method": "multi-layer per-layer ortho", "layer_range": args.layer_range,
            "strength": args.strength, "n_dirs_used_embed": len(selected_dirs),
            "aucs": dict(aucs)}
    (out / "abliterate_meta.json").write_text(json.dumps(meta, indent=2))
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
