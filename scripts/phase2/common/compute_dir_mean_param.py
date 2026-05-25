"""Phase 2 MEAN-direction computer, parametrized on --run-dir."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def auc(pos, neg):
    n_p, n_n = len(pos), len(neg)
    wins = sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg)
    return wins / max(1, n_p * n_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--acts", default=None)
    args = ap.parse_args()

    run = Path(args.run_dir)
    acts = Path(args.acts) if args.acts else run / "activations" / "pf_acts.npz"
    z = np.load(acts)
    refuse = z["refuse"]; comply = z["comply"]
    print(f"refuse={refuse.shape}  comply={comply.shape}")
    n_layers = refuse.shape[1] - 1
    d_model = refuse.shape[2]
    print(f"n_layers={n_layers} d_model={d_model}")

    rows = []
    for L in range(1, n_layers + 1):
        mu_r = refuse[:, L].mean(0)
        mu_c = comply[:, L].mean(0)
        raw = mu_r - mu_c
        n = float(np.linalg.norm(raw))
        if n == 0:
            continue
        d = raw / n
        s_r = refuse[:, L] @ d
        s_c = comply[:, L] @ d
        a = auc(s_r, s_c)
        margin = float(s_r.mean() - s_c.mean())
        rows.append((L, a, margin, n, d))
        print(f"  L={L:2d}  AUC={a:.3f}  margin={margin:+.3f}  ||d||={n:.2f}")

    rows.sort(key=lambda r: abs(r[1] - 0.5), reverse=True)
    best = rows[0]
    print(f"\nbest layer = L={best[0]}  AUC={best[1]:.3f}  margin={best[2]:+.3f}")

    L, a, margin, norm, d_vec = best
    out_dir = run / "vectors"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"pf_dir_L{L}.pt"
    torch.save({"layer": L, "dir": torch.from_numpy(d_vec.astype(np.float32)),
                "auc": a, "margin": margin, "norm": norm}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
