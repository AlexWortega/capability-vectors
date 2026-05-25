"""Insert a new variant row into the CLAUDE.md results table.

Looks for the markdown table after `## Variants and results` and adds a row
just before the line that starts with `**Key findings**`. Idempotent on
(variant, exp) — replaces existing matching row.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claudemd", default="CLAUDE.md")
    ap.add_argument("--variant", required=True, help="e.g. exp5_agentonly")
    ap.add_argument("--exp", default="phase2", help="short label for # column")
    ap.add_argument("--contrast", default="—")
    ap.add_argument("--method", default="—")
    ap.add_argument("--layer", default="—")
    ap.add_argument("--auc", default="—")
    ap.add_argument("--strength", default="0.5")
    ap.add_argument("--tbench", default="—")
    ap.add_argument("--ha20", default="—")
    ap.add_argument("--mmlu", default="—")
    ap.add_argument("--eq", default="—")
    ap.add_argument("--hf-url", default="")
    args = ap.parse_args()

    path = Path(args.claudemd)
    text = path.read_text()
    name = args.variant
    if args.hf_url:
        first_col = f"[`..-abliterated-{name}`]({args.hf_url})"
    else:
        first_col = f"`..-abliterated-{name}` (not pushed)"

    new_row = (
        f"| {args.exp} | {first_col} | {args.contrast} | {args.method} | "
        f"{args.layer} | {args.auc} | {args.strength} | {args.tbench} | "
        f"{args.ha20} | {args.mmlu} | {args.eq} |"
    )

    lines = text.splitlines(keepends=True)
    out = []
    in_table = False
    appended = False
    for i, ln in enumerate(lines):
        if ln.startswith("| #"):
            in_table = True
        if in_table and not appended:
            # Replace existing row for same variant if present
            if name in ln and ln.startswith("|"):
                out.append(new_row + "\n")
                appended = True
                continue
            if ln.startswith("**Key findings"):
                out.append(new_row + "\n")
                out.append("\n")
                out.append(ln)
                appended = True
                in_table = False
                continue
        out.append(ln)
    if not appended:
        out.append("\n" + new_row + "\n")
    path.write_text("".join(out))
    print(f"updated {path} with row for {name}")


if __name__ == "__main__":
    main()
