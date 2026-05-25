"""Compute pass-vs-fail direction per layer, choose best by AUC."""
import numpy as np
import torch
from pathlib import Path

RUN = Path("/workspace/capvec_pf")
z = np.load(RUN / "activations" / "pf_acts.npz")
refuse = z["refuse"]   # FAIL trajectories  [N, L+1, D]
comply = z["comply"]   # PASS trajectories
print(f"refuse(FAIL)={refuse.shape}  comply(PASS)={comply.shape}")
n_layers = refuse.shape[1] - 1
d_model  = refuse.shape[2]
print(f"n_layers={n_layers} d_model={d_model}")

def auc(pos, neg):
    n_p, n_n = len(pos), len(neg)
    wins = sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg)
    return wins / (n_p * n_n)

rows = []
for L in range(1, n_layers + 1):
    mu_r = refuse[:, L].mean(0)
    mu_c = comply[:, L].mean(0)
    raw = mu_r - mu_c          # FAIL - PASS direction
    n = np.linalg.norm(raw)
    if n == 0: continue
    d = raw / n
    # project each example onto d → score
    s_r = refuse[:, L] @ d     # [N]
    s_c = comply[:, L] @ d
    auc_l = auc(s_r, s_c)      # how well d separates FAIL from PASS
    margin = float(s_r.mean() - s_c.mean())
    rows.append((L, auc_l, margin, n))
    print(f"  L={L:2d}  AUC={auc_l:.3f}  margin={margin:+.3f}  ||d||={n:.2f}")

# best: max(|AUC - 0.5|) for strong separation
rows.sort(key=lambda r: abs(r[1] - 0.5), reverse=True)
best = rows[0]
print(f"\nbest layer = L={best[0]}  AUC={best[1]:.3f}  margin={best[2]:+.3f}")

# Save chosen direction
L = best[0]
mu_r = refuse[:, L].mean(0)
mu_c = comply[:, L].mean(0)
raw = mu_r - mu_c
d = raw / np.linalg.norm(raw)
out = RUN / "vectors" / f"pf_dir_L{L}.pt"
out.parent.mkdir(exist_ok=True)
torch.save({"layer": L, "dir": torch.from_numpy(d), "auc": best[1], "margin": best[2], "norm": float(best[3])}, out)
print(f"saved -> {out}")
