"""exp1 — direction from multi-position activations.

Two aggregators (per layer):
  (a) mean-across-positions:  μ_FAIL - μ_PASS  averaged over positions, then normalised
  (b) concat-positions + top-1 PCA: stack diff vectors at every (position,layer),
      take SVD top component

For each layer we evaluate AUC using the SAME aggregation: project each
trajectory's mean-position residual onto d, separate PASS/FAIL.

Picks best (layer, method) by max |AUC - 0.5|, saves to vectors/pf_dir_multi_<method>_L<N>.pt.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def auc_score(pos, neg):
    n_p, n_n = len(pos), len(neg)
    wins = sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg)
    return wins / max(1, n_p * n_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/workspace/capvec_pf_v2")
    ap.add_argument("--acts", default=None,
                    help="Path to multi-token npz; default <run-dir>/activations/pf_acts_multi.npz")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    acts_path = Path(args.acts) if args.acts else run_dir / "activations" / "pf_acts_multi.npz"
    z = np.load(acts_path)
    refuse = z["refuse"]   # [N, P, L+1, D]
    comply = z["comply"]
    print(f"refuse={refuse.shape}  comply={comply.shape}")
    n_pos = refuse.shape[1]
    n_layers = refuse.shape[2] - 1
    d_model = refuse.shape[3]
    print(f"n_pos={n_pos} n_layers={n_layers} d_model={d_model}")

    rows = []   # (layer, method, auc, margin, norm, direction_vec)

    for L in range(1, n_layers + 1):
        r_lp = refuse[:, :, L, :]   # [N, P, D]
        c_lp = comply[:, :, L, :]
        r_mean = r_lp.mean(0)        # [P, D]
        c_mean = c_lp.mean(0)
        diff_pos = r_mean - c_mean   # [P, D]

        # --- method (a) MEAN across positions ---
        d_mean = diff_pos.mean(0)
        nm = float(np.linalg.norm(d_mean))
        if nm > 0:
            d_a = d_mean / nm
            r_score = r_lp.mean(1) @ d_a      # mean over positions per traj  [N]
            c_score = c_lp.mean(1) @ d_a
            a = auc_score(r_score, c_score)
            margin = float(r_score.mean() - c_score.mean())
            rows.append((L, "mean", a, margin, nm, d_a))

        # --- method (b) PCA top-1 across positions ---
        # Stack per-trajectory paired diffs:  refuse_traj_mean - comply_traj_mean is not paired,
        # so use the per-(N,P) centered diff matrix:
        diff_stack = (r_lp - r_lp.mean(0, keepdims=True)) - (c_lp - c_lp.mean(0, keepdims=True))
        diff_stack = diff_stack.reshape(-1, d_model)   # [N*P, D]
        try:
            # SVD on small mat OK; D=2560, N*P ≈ 60*5=300
            u, s, vt = np.linalg.svd(diff_stack.astype(np.float32), full_matrices=False)
            d_b = vt[0]
            # sign so PASS - FAIL projection is consistent w/ method-a
            ref_sign = float(diff_pos.mean(0) @ d_b)
            if ref_sign < 0:
                d_b = -d_b
            d_b = d_b / max(1e-9, np.linalg.norm(d_b))
            r_score = r_lp.mean(1) @ d_b
            c_score = c_lp.mean(1) @ d_b
            a = auc_score(r_score, c_score)
            margin = float(r_score.mean() - c_score.mean())
            rows.append((L, "pca", a, margin, float(s[0]), d_b))
        except Exception as e:
            print(f"  L={L} pca skip: {e}")

        print(f"  L={L:2d}  mean AUC={rows[-2][2]:.3f} margin={rows[-2][3]:+.3f}  ||d||={rows[-2][4]:.2f}    "
              f"pca AUC={rows[-1][2]:.3f} margin={rows[-1][3]:+.3f} sv1={rows[-1][4]:.2f}")

    rows.sort(key=lambda r: abs(r[2] - 0.5), reverse=True)
    best = rows[0]
    print(f"\nbest: L={best[0]} method={best[1]} AUC={best[2]:.3f} margin={best[3]:+.3f}")

    out_dir = run_dir / "vectors"
    out_dir.mkdir(exist_ok=True)
    L, method, auc, margin, norm, d_vec = best
    out = out_dir / f"pf_dir_multi_{method}_L{L}.pt"
    torch.save({"layer": L, "dir": torch.from_numpy(d_vec.astype(np.float32)),
                "auc": auc, "margin": margin, "norm": norm,
                "method": f"multi-token-{method}"}, out)
    print(f"saved -> {out}")

    # Also save runner-up of the other method type for inspection
    other_method = "pca" if method == "mean" else "mean"
    for r in rows:
        if r[1] == other_method:
            L2, m2, a2, mg2, n2, d2 = r
            out2 = out_dir / f"pf_dir_multi_{m2}_L{L2}.pt"
            torch.save({"layer": L2, "dir": torch.from_numpy(d2.astype(np.float32)),
                        "auc": a2, "margin": mg2, "norm": n2,
                        "method": f"multi-token-{m2}"}, out2)
            print(f"runner-up saved -> {out2}")
            break


if __name__ == "__main__":
    main()
