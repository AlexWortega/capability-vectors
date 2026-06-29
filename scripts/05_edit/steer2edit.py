"""Steer2Edit: closed-form rank-1 weight edit along a steering direction.

Implements Theorems 3.1-3.3 from "Steer2Edit: Efficient Editing via Steering
Vectors and Weight Edit Decomposition" (arxiv:2602.09870).

Unlike abliterate_single_layer.py (which orthogonalizes ALL residual writers),
this script:
  1. Computes a rank-1 perturbation delta_W = lambda * u @ k^T for each weight
     matrix, where:
     - u  = output direction (unit vector in output space aligned with v)
     - k  = input direction (how W currently contributes to v)
     - λ  = elastic-net regularised magnitude (Theorem 3.3)
  2. Only edits weights where |cos(W @ k, v)| exceeds a selectivity threshold
     (rho), leaving other weights unchanged.
  3. Blends the edit with `strength α`: W_new = W - α * delta_W

Usage:
    python scripts/05_edit/steer2edit.py \
        --dir-pt /workspace/trajedit/vectors/pf_dir_trajedit_L16.pt \
        --model-path Qwen/Qwen3-8B-Instruct \
        --out /workspace/trajedit_v1 \
        --strength 0.5 \
        --rho 0.5 \
        --alpha-en 0.5

Produces a standalone model at --out (same save_pretrained format as abliterate).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def steer2edit_delta(
    W: torch.Tensor,   # [D_out, D_in]
    v: torch.Tensor,   # [D_out] — output steering direction (unit)
    rho: float = 0.5,  # sparsity threshold (elastic-net)
    alpha_en: float = 0.5,  # elastic-net balance (0=L1, 1=L2)
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the rank-1 delta matrix delta_W = λ * u ⊗ k^hat.

    Theorem 3.1: k_hat = W^T v / ||W^T v||  (input direction)
    Theorem 3.2: u_hat = v / ||v||            (output direction — already unit)
    Theorem 3.3: λ via elastic-net shrinkage

    Returns delta_W [D_out, D_in] with the same dtype/device as W.
    """
    orig_dtype = W.dtype
    W_f = W.float()
    v_f = v.float()

    # Input direction (Theorem 3.1)
    k = W_f.T @ v_f           # [D_in]
    k_norm = k.norm()
    if k_norm < eps:
        return torch.zeros_like(W)
    k_hat = k / k_norm         # [D_in]

    # Output direction (Theorem 3.2) — v is already a unit vector from compute_dir
    u_hat = v_f / (v_f.norm() + eps)  # [D_out]

    # Effective alignment: how strongly W maps k_hat onto v
    # g = v^T (W k_hat) / ||v|| = cos-alignment between W k_hat and v
    Wk = W_f @ k_hat           # [D_out]
    g = float(v_f @ Wk) / (float(v_f.norm()) * float(Wk.norm()) + eps)

    # Elastic-net magnitude (Theorem 3.3)
    # λ = sign(g) * max(0, |g| - rho*alpha_en) / (rho * (1 - alpha_en))
    denom = rho * (1.0 - alpha_en)
    if denom < eps:
        # Degenerate: L1 only → hard threshold
        lam = 0.0 if abs(g) <= rho * alpha_en else (abs(g) - rho * alpha_en) * (1 if g > 0 else -1)
    else:
        lam = (abs(g) - rho * alpha_en)
        lam = max(0.0, lam) / denom * (1 if g > 0 else -1)

    delta_W = lam * u_hat.unsqueeze(1) * k_hat.unsqueeze(0)  # [D_out, D_in]
    return delta_W.to(dtype=orig_dtype)


def find_writers(model) -> list[tuple[str, torch.nn.Module]]:
    """Return (name, module) for all residual-writing projections."""
    writers = []
    for li, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            writers.append((f"L{li}.self_attn.o_proj", layer.self_attn.o_proj))
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            writers.append((f"L{li}.mlp.down_proj", layer.mlp.down_proj))
    return writers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-pt", required=True,
                    help="Direction .pt file (output of compute_trajedit_dir.py)")
    ap.add_argument("--model-path", default="Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--out", required=True, help="Output model directory")
    ap.add_argument("--strength", type=float, default=0.5,
                    help="Blend factor α: W_new = W - α * delta_W")
    ap.add_argument("--rho", type=float, default=0.5,
                    help="Elastic-net sparsity threshold")
    ap.add_argument("--alpha-en", type=float, default=0.5,
                    help="Elastic-net balance (0=L1, 1=L2)")
    ap.add_argument("--edit-embed", action="store_true", default=True,
                    help="Also edit embed_tokens rows (default: True)")
    args = ap.parse_args()

    blob = torch.load(args.dir_pt, weights_only=False, map_location="cpu")
    L = blob["layer"]
    r = blob["dir"].float()  # [D] unit vector, fail-minus-pass direction
    method = blob.get("method", "unknown")
    print(
        f"[direction] L={L}  AUC={blob.get('auc',0):.3f}  "
        f"margin={blob.get('margin',0):+.3f}  method={method}",
        flush=True,
    )

    print(f"[load] {args.model_path}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    print(f"[load] took={time.time()-t0:.1f}s", flush=True)

    r = r.to(dtype=torch.float32)
    edited_count = 0

    # Edit embed_tokens rows (output direction = embedding vectors)
    if args.edit_embed:
        E = model.model.embed_tokens.weight.data.float()  # [vocab, D]
        # For embed_tokens: each row is a D-dim vector; edit as if W=[1,D] for each row
        # Simpler: use abliteration-style row ortho scaled by strength (rank-1 approx)
        # delta = proj_mag * r^T; proj_mag = E @ r per row
        proj_mag = E @ r               # [vocab]
        delta = proj_mag.unsqueeze(1) * r.unsqueeze(0)   # [vocab, D]
        E_new = E - args.strength * delta
        model.model.embed_tokens.weight.data.copy_(E_new.to(torch.bfloat16))
        edited_count += 1
        print(f"[edit] embed_tokens rows orthogonalized (strength={args.strength})", flush=True)

    writers = find_writers(model)
    print(f"[edit] {len(writers)} residual-writer projections", flush=True)

    alignment_stats = []
    with torch.no_grad():
        for name, mod in writers:
            W = mod.weight.data.float()
            delta_W = steer2edit_delta(W, r, rho=args.rho, alpha_en=args.alpha_en)
            W_new = W - args.strength * delta_W
            mod.weight.data.copy_(W_new.to(torch.bfloat16))

            # Track alignment magnitude for diagnostics
            lam_mag = float(delta_W.norm()) / (W.numel() ** 0.5 + 1e-8)
            alignment_stats.append({"name": name, "delta_frob_norm_per_param": lam_mag})
            edited_count += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[save] → {out}", flush=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)

    meta = {
        "source_model": args.model_path,
        "direction_pt": str(args.dir_pt),
        "direction_layer": L,
        "direction_auc": float(blob.get("auc", 0)),
        "direction_method": method,
        "strength": args.strength,
        "rho": args.rho,
        "alpha_en": args.alpha_en,
        "n_edited": edited_count,
        "edit_method": "steer2edit_rank1",
        "alignment_stats": alignment_stats[:5],  # top-5 for brevity
    }
    (out / "trajedit_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] edited {edited_count} weight matrices  meta → {out}/trajedit_meta.json", flush=True)
    print(f"\nNext step — wrap for sglang:", flush=True)
    print(f"  python scripts/06_wrap_vlm/wrap_text_to_vlm.py --in {out}", flush=True)


if __name__ == "__main__":
    main()
