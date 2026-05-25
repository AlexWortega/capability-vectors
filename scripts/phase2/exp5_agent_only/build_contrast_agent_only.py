"""exp5 — build agent-only contrast.

Drops every row whose source starts with `mmlu-pi` from BOTH PASS and FAIL
buckets of rift_dataset_v2, then samples up to N PASS + N FAIL from the
remaining (claw-eval-*, tbench-2-*).

Output mirrors capvec_pf_v2 layout but at /workspace/capvec_pf_v7_agentonly/.
"""
from __future__ import annotations
import json, os, random
from collections import Counter

random.seed(43)

ROOT = "/workspace"
TRAIN = ROOT + "/rift_dataset_v2/train.jsonl"
CLEAN = ROOT + "/clean_fails.jsonl"
OUT = ROOT + "/capvec_pf_v7_agentonly"
os.makedirs(OUT, exist_ok=True)
N = 60
MAX_CHARS = 12000


def is_agent_src(src: str) -> bool:
    """Anything that is not an mmlu-pi-* trajectory."""
    return not src.startswith("mmlu-pi")


def render(t):
    parts = [f"<|{m['role']}|>\n{m['content']}" for m in t["messages"]]
    s = "\n".join(parts)
    if len(s) > MAX_CHARS:
        s = s[-MAX_CHARS:]
    return s


def main():
    rows = [json.loads(l) for l in open(TRAIN)]
    print(f"total rows: {len(rows)}")
    print("by source (raw):", Counter(r["source"] for r in rows))

    pos_all = [r for r in rows if r.get("reward") == 1.0 and is_agent_src(r["source"])]
    print(f"PASS pool after agent-only filter: {len(pos_all)}")

    # Clean FAIL bucket (gemini-labeled) — filter to agent-only too
    if os.path.exists(CLEAN):
        clean_negs = [json.loads(l) for l in open(CLEAN)
                      if json.loads(l).get("label") == "CLEAN_FAIL"]
        clean_negs = [r for r in clean_negs if is_agent_src(r["source"])]
        print(f"CLEAN_FAIL agent-only pool: {len(clean_negs)}")
    else:
        clean_negs = [r for r in rows if r.get("reward") == 0.0 and is_agent_src(r["source"])]
        print(f"[warn] no clean_fails.jsonl, using raw FAIL pool: {len(clean_negs)}")

    if not pos_all or not clean_negs:
        raise SystemExit(f"empty bucket: pos={len(pos_all)} neg={len(clean_negs)}")

    random.shuffle(pos_all)
    random.shuffle(clean_negs)

    pos_pick = pos_all[:N]
    neg_pick = clean_negs[:N]
    print(f"picked PASS={len(pos_pick)}  FAIL={len(neg_pick)}")
    print("PASS sources:", Counter(r["source"] for r in pos_pick))
    print("FAIL sources:", Counter(r["source"] for r in neg_pick))

    with open(f"{OUT}/contrast_comply.jsonl", "w") as f:
        for r in pos_pick:
            f.write(json.dumps({"prompt": render(r), "source": r["source"],
                                "task_id": r.get("task_id"),
                                "reward": r.get("reward")}, ensure_ascii=False) + "\n")
    with open(f"{OUT}/contrast_refuse.jsonl", "w") as f:
        for r in neg_pick:
            f.write(json.dumps({"prompt": render(r), "source": r["source"],
                                "task_id": r.get("task_id"),
                                "reason": r.get("reason", "")}, ensure_ascii=False) + "\n")
    print(f"wrote -> {OUT}/contrast_{{comply,refuse}}.jsonl")


if __name__ == "__main__":
    main()
