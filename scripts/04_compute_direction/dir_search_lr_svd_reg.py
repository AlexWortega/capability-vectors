"""Phase 1: compute 3 alternative directions from soyuz-only activations.
  A) MEAN diff (baseline — same as v4)
  B) LR — logistic regression per layer; weight vector = direction
  C) SVD — top singular vector of (refuse - comply) point cloud
  D) REG — continuous reward regression (rewards used as labels, currently binary 0/1)

For each: choose best layer by AUC, save .pt for abliterate.
"""
import numpy as np, json, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge

RUN = Path("/workspace/capvec_pf_v4_soyuzonly")
z = np.load(RUN / "activations" / "pf_acts.npz")
refuse = z["refuse"]  # FAIL [N,L+1,D]
comply = z["comply"]  # PASS
n_layers = refuse.shape[1] - 1
print(f"refuse={refuse.shape} comply={comply.shape}")

X_per_layer = []  # list of (X, y) per layer L=1..L_max
for L in range(1, n_layers+1):
    X = np.concatenate([refuse[:, L], comply[:, L]], axis=0)  # [2N, D]
    y = np.concatenate([np.ones(len(refuse)), np.zeros(len(comply))])  # FAIL=1, PASS=0
    X_per_layer.append((L, X, y))

def auc(scores, labels):
    pos = scores[labels==1]; neg = scores[labels==0]
    np_p, np_n = len(pos), len(neg)
    wins = sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg)
    return wins/(np_p*np_n)

# A) MEAN
print("\n--- A: MEAN diff ---")
A_rows = []
for L, X, y in X_per_layer:
    d = (X[y==1].mean(0) - X[y==0].mean(0))
    d = d / np.linalg.norm(d)
    s = X @ d; a = auc(s, y)
    A_rows.append((L, a, d))
bestA = max(A_rows, key=lambda r: abs(r[1] - 0.5))
print(f"best L={bestA[0]} AUC={bestA[1]:.3f}")

# B) LR probe
print("\n--- B: LR probe ---")
B_rows = []
for L, X, y in X_per_layer:
    clf = LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")
    clf.fit(X.astype(np.float32), y)
    w = clf.coef_[0]; w = w / np.linalg.norm(w)
    s = X @ w; a = auc(s, y)
    B_rows.append((L, a, w))
bestB = max(B_rows, key=lambda r: abs(r[1] - 0.5))
print(f"best L={bestB[0]} AUC={bestB[1]:.3f}")

# C) SVD top-1 of diff matrix
print("\n--- C: SVD top-1 ---")
C_rows = []
for L, X, y in X_per_layer:
    diff = X[y==1] - X[y==0][:len(X[y==1])]  # paired diff (truncate to min len)
    U, S, Vt = np.linalg.svd(diff.astype(np.float32), full_matrices=False)
    d = Vt[0]; d = d / np.linalg.norm(d)
    if (X[y==1] @ d).mean() < (X[y==0] @ d).mean(): d = -d
    s = X @ d; a = auc(s, y)
    C_rows.append((L, a, d))
bestC = max(C_rows, key=lambda r: abs(r[1] - 0.5))
print(f"best L={bestC[0]} AUC={bestC[1]:.3f}")

# D) Reward regression
print("\n--- D: Reward regression ---")
D_rows = []
for L, X, y in X_per_layer:
    # y is binary 0/1 here (same data); for fairer regression use {-1,+1}? Or keep
    targets = (-y).astype(np.float32) + 0.5  # FAIL=-0.5, PASS=+0.5
    reg = Ridge(alpha=1.0)
    reg.fit(X.astype(np.float32), targets)
    w = reg.coef_; w = w / np.linalg.norm(w)
    # direction = "toward PASS" — flip if needed (we want refuse>comply on scores)
    if (X[y==1] @ w).mean() < (X[y==0] @ w).mean(): w = -w
    s = X @ w; a = auc(s, y)
    D_rows.append((L, a, w))
bestD = max(D_rows, key=lambda r: abs(r[1] - 0.5))
print(f"best L={bestD[0]} AUC={bestD[1]:.3f}")

# Save best of each method
out_dir = RUN / "vectors_v5"
out_dir.mkdir(exist_ok=True)
for name, best in [("MEAN", bestA), ("LR", bestB), ("SVD", bestC), ("REG", bestD)]:
    L, a, d = best
    torch.save({"layer": L, "dir": torch.from_numpy(d.astype(np.float32)),
                "auc": float(a), "method": name},
               out_dir / f"v5_{name}_L{L}.pt")
    print(f"saved {name} L={L} AUC={a:.3f} -> {out_dir}/v5_{name}_L{L}.pt")

print("\nSummary:")
print(f"  MEAN best: L={bestA[0]} AUC={bestA[1]:.3f}")
print(f"  LR   best: L={bestB[0]} AUC={bestB[1]:.3f}")
print(f"  SVD  best: L={bestC[0]} AUC={bestC[1]:.3f}")
print(f"  REG  best: L={bestD[0]} AUC={bestD[1]:.3f}")
