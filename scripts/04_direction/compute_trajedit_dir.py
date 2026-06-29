"""Compute TrajEdit direction and ablation-comparison directions.

Two directions are computed side by side to enable the "localization matters" ablation:
  - TrajEdit direction: mean(fail_firsterror) - mean(pass_recovery) at each layer
  - Outcome-only direction: mean(fail_outcome) - mean(pass_outcome)  [current pipeline]

Both are normalized. AUC is computed for each layer to pick the best layer.

Output:
    vectors/pf_dir_trajedit_L{n}.pt   ← TrajEdit direction at each layer
    vectors/pf_dir_outcome_L{n}.pt    ← outcome-only (for Control-A ablation)
    layer_auc_comparison.json         ← layer vs AUC for both methods

Usage:
    python scripts/04_direction/compute_trajedit_dir.py \
        --acts-dir /workspace/trajedit/activations \
        --out-dir /workspace/trajedit/vectors
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    n_p, n_n = len(pos), len(neg)
    wins = sum(
        1 if p > n else 0.5 if p == n else 0
        for p in pos for n in neg
    )
    return wins / (n_p * n_n)


def compute_direction(fail: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, float]:
    """Mean-difference direction, return (unit_vector, AUC)."""
    mu_fail = fail.mean(0)
    mu_correct = correct.mean(0)
    raw = mu_fail - mu_correct
    norm = np.linalg.norm(raw)
    if norm < 1e-12:
        return np.zeros_like(raw), 0.5
    d = raw / norm
    s_fail = fail @ d
    s_correct = correct @ d
    a = auc(s_fail, s_correct)
    # Canonical: AUC should be >0.5 (fail scores higher on d)
    if a < 0.5:
        d = -d
        a = 1.0 - a
    return d, a


def sweep_layers(
    fail_arr: np.ndarray,
    correct_arr: np.ndarray,
    label: str,
) -> list[dict]:
    """Sweep all layers, return sorted list of {layer, auc, margin}."""
    n_layers = fail_arr.shape[1] - 1  # shape [N, L+1, D], layer 0 = embedding
    rows = []
    for L in range(1, n_layers + 1):
        d, a = compute_direction(fail_arr[:, L], correct_arr[:, L])
        margin = float((fail_arr[:, L] @ d).mean() - (correct_arr[:, L] @ d).mean())
        rows.append({"layer": L, "auc": float(a), "margin": float(margin), "dir": d})
        print(f"  [{label}] L={L:2d}  AUC={a:.3f}  margin={margin:+.3f}", flush=True)
    return rows


def pick_best(rows: list[dict]) -> dict:
    """Best layer: max |AUC - 0.5| (strongest separation, excluding overfit AUC=1)."""
    # Penalize AUC=1 (likely overfitting with tiny N)
    def score(r):
        a = r["auc"]
        if a >= 0.999:
            return 0.0  # skip degenerate
        return abs(a - 0.5)
    return max(rows, key=score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="/workspace/trajedit/activations")
    ap.add_argument("--out-dir", default="/workspace/trajedit/vectors")
    args = ap.parse_args()

    acts_dir = Path(args.acts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] activations", flush=True)
    fail_fe = np.load(acts_dir / "fail_firsterror.npy")   # [N, L+1, D]
    pass_rc = np.load(acts_dir / "pass_recovery.npy")
    fail_oc = np.load(acts_dir / "fail_outcome.npy")
    pass_oc = np.load(acts_dir / "pass_outcome.npy")

    N_fail, nl1, D = fail_fe.shape
    N_pass = pass_rc.shape[0]
    n_layers = nl1 - 1
    print(f"fail_firsterror={fail_fe.shape}  pass_recovery={pass_rc.shape}", flush=True)
    print(f"N_fail={N_fail} N_pass={N_pass} n_layers={n_layers} D={D}", flush=True)

    print("\n[TrajEdit direction sweep]", flush=True)
    te_rows = sweep_layers(fail_fe, pass_rc, "TrajEdit")
    best_te = pick_best(te_rows)
    print(f"\n[TrajEdit] best L={best_te['layer']}  AUC={best_te['auc']:.3f}", flush=True)

    print("\n[Outcome-only direction sweep]", flush=True)
    oc_rows = sweep_layers(fail_oc, pass_oc, "Outcome")
    best_oc = pick_best(oc_rows)
    print(f"\n[Outcome-only] best L={best_oc['layer']}  AUC={best_oc['auc']:.3f}", flush=True)

    # Save all layer directions
    for row in te_rows:
        L = row["layer"]
        d = torch.from_numpy(row["dir"].astype(np.float32))
        torch.save({
            "layer": L, "dir": d, "auc": row["auc"], "margin": row["margin"],
            "method": "trajedit_mean_firsterror_recovery",
            "N_fail": N_fail, "N_pass": N_pass,
        }, out_dir / f"pf_dir_trajedit_L{L}.pt")

    for row in oc_rows:
        L = row["layer"]
        d = torch.from_numpy(row["dir"].astype(np.float32))
        torch.save({
            "layer": L, "dir": d, "auc": row["auc"], "margin": row["margin"],
            "method": "outcome_only_mean",
            "N_fail": N_fail, "N_pass": N_pass,
        }, out_dir / f"pf_dir_outcome_L{L}.pt")

    # Save comparison table
    comparison = {
        "trajedit": [
            {"layer": r["layer"], "auc": r["auc"], "margin": r["margin"]}
            for r in te_rows
        ],
        "outcome_only": [
            {"layer": r["layer"], "auc": r["auc"], "margin": r["margin"]}
            for r in oc_rows
        ],
        "best_trajedit": {"layer": best_te["layer"], "auc": best_te["auc"]},
        "best_outcome_only": {"layer": best_oc["layer"], "auc": best_oc["auc"]},
    }
    (out_dir / "layer_auc_comparison.json").write_text(json.dumps(comparison, indent=2))

    print(f"\n[done] saved {len(te_rows)*2} direction files + comparison table to {out_dir}", flush=True)
    print(f"  Best TrajEdit:     L={best_te['layer']}  AUC={best_te['auc']:.3f}", flush=True)
    print(f"  Best Outcome-only: L={best_oc['layer']}  AUC={best_oc['auc']:.3f}", flush=True)
    print(f"\n  Next step (TrajEdit-a):", flush=True)
    print(f"    python scripts/05_edit/steer2edit.py \\", flush=True)
    print(f"        --dir-pt {out_dir}/pf_dir_trajedit_L{best_te['layer']}.pt \\", flush=True)
    print(f"        --out /workspace/trajedit_v1  --strength 0.5", flush=True)
    print(f"\n  Next step (Control-A — current abliterate for comparison):", flush=True)
    print(f"    python scripts/05_abliterate/abliterate_single_layer.py \\", flush=True)
    print(f"        --dir-pt {out_dir}/pf_dir_outcome_L{best_oc['layer']}.pt \\", flush=True)
    print(f"        --out /workspace/control_a  --strength 0.5", flush=True)


if __name__ == "__main__":
    main()
