"""Append a result row to results/all_variants.csv (idempotent)."""
from __future__ import annotations
import argparse, csv, os, datetime
from pathlib import Path

HEADER = ["ts", "exp", "variant", "tbench_pass", "tbench_total",
          "ha20_pass", "ha20_total", "mmlu_pro", "eqbench",
          "auc", "layer", "method", "strength", "hf_url", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tbench-pass", type=int, default=0)
    ap.add_argument("--tbench-total", type=int, default=17)
    ap.add_argument("--ha20-pass", type=int, default=0)
    ap.add_argument("--ha20-total", type=int, default=20)
    ap.add_argument("--mmlu-pro", default="")
    ap.add_argument("--eqbench", default="")
    ap.add_argument("--auc", default="")
    ap.add_argument("--layer", default="")
    ap.add_argument("--method", default="mean")
    ap.add_argument("--strength", default="0.5")
    ap.add_argument("--hf-url", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    out = Path(args.csv); out.parent.mkdir(exist_ok=True, parents=True)
    new_file = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        w.writerow([
            datetime.datetime.utcnow().isoformat() + "Z",
            args.exp, args.variant,
            args.tbench_pass, args.tbench_total,
            args.ha20_pass, args.ha20_total,
            args.mmlu_pro, args.eqbench,
            args.auc, args.layer, args.method, args.strength,
            args.hf_url, args.notes,
        ])
    print(f"appended -> {out}")


if __name__ == "__main__":
    main()
