"""Find recovery steps in PASS traces.

A "recovery step" is the turn where the model encountered an error (tool returned
an error) but the trace ultimately succeeded. We want the residual at the turn
BEFORE the recovery (i.e., the last error-containing turn), which captures the
"about to recover" representation.

Heuristic priority:
  1. Last tool result that contains an error keyword before the final success.
  2. If no error in tool results: second-to-last assistant turn (the model was
     still working, not yet at the final answer).

Usage:
    python scripts/02_localize/find_recovery.py \
        --rollouts /workspace/trajedit/rollouts_localized.jsonl \
        --out /workspace/trajedit/rollouts_with_recovery.jsonl
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ERROR_KEYWORDS = (
    "error", "exception", "traceback", "failed", "failure",
    "not found", "no such", "permission denied", "timeout",
    "undefined", "typeerror", "nameerror", "valueerror",
)


def find_recovery_step(turns: list[dict]) -> int | None:
    """Return the turn index (in the full turns list) of the recovery step.

    Returns the index of the last tool-result turn with an error keyword that
    still precedes the final successful assistant turn. Returns None if no such
    turn exists.
    """
    asst_indices = [i for i, t in enumerate(turns) if t["role"] == "assistant"]
    if len(asst_indices) < 2:
        return None

    # Heuristic 1: find the last error-containing tool result
    last_error_idx: int | None = None
    for i, t in enumerate(turns):
        if t["role"] == "tool":
            content = (t.get("content") or "").lower()
            if any(kw in content for kw in ERROR_KEYWORDS):
                # Only count if it's NOT the last turn group
                last_asst = asst_indices[-1]
                if i < last_asst:
                    last_error_idx = i

    if last_error_idx is not None:
        return last_error_idx

    # Heuristic 2: second-to-last assistant turn
    if len(asst_indices) >= 2:
        return asst_indices[-2]

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True,
                    help="JSONL with localized traces (output of prefix_flip.py)")
    ap.add_argument("--out", required=True,
                    help="JSONL with recovery_step field added to PASS traces")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pass = n_fail = n_recovery_found = 0
    with open(args.rollouts) as fin, open(out_path, "w") as fout:
        for line in fin:
            trace = json.loads(line)
            if trace["outcome"] == "pass":
                step = find_recovery_step(trace["turns"])
                trace["recovery_step"] = step
                n_pass += 1
                if step is not None:
                    n_recovery_found += 1
                print(
                    f"[pass] {trace['task_id']:30s} n_turns={trace['n_turns']} "
                    f"recovery_step={step}",
                    flush=True,
                )
            else:
                trace.setdefault("recovery_step", None)
                n_fail += 1

            fout.write(json.dumps(trace) + "\n")

    print(
        f"\n[done] pass={n_pass} (recovery found={n_recovery_found})  fail={n_fail}"
        f"  → {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
