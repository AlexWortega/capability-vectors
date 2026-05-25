"""Build contrast v2: same pristine PASS bucket + cleaner FAIL bucket (Gemini-filtered)."""
import json, os, random
random.seed(43)

ROOT = "/workspace"   # eva02 host equivalent
CLEAN = ROOT + "/clean_fails.jsonl"
TRAIN = ROOT + "/rift_dataset_v2/train.jsonl"
OUT = ROOT + "/capvec_pf_v2"
os.makedirs(OUT, exist_ok=True)
N = 60
MAX_CHARS = 12000

# Reuse same PASS pool
rows = [json.loads(l) for l in open(TRAIN)]
pos = [r for r in rows if r["reward"] == 1.0]
clean_negs = [json.loads(l) for l in open(CLEAN) if json.loads(l)["label"] == "CLEAN_FAIL"]
print(f"pos pool={len(pos)}  clean_neg pool={len(clean_negs)}")

def render(t):
    parts = []
    for m in t["messages"]:
        parts.append(f"<|{m['role']}|>\n{m['content']}")
    s = "\n".join(parts)
    if len(s) > MAX_CHARS: s = s[-MAX_CHARS:]
    return s

random.shuffle(pos); random.shuffle(clean_negs)
pos_pick = pos[:N]; neg_pick = clean_negs[:N]
print(f"picked pos={len(pos_pick)} clean_neg={len(neg_pick)}")

with open(f"{OUT}/contrast_comply.jsonl", "w") as f:
    for r in pos_pick: f.write(json.dumps({"prompt": render(r), "source": r["source"], "task_id": r["task_id"], "reward": r["reward"]}, ensure_ascii=False) + "\n")
with open(f"{OUT}/contrast_refuse.jsonl", "w") as f:
    for r in neg_pick: f.write(json.dumps({"prompt": render(r), "source": r["source"], "task_id": r["task_id"], "reason": r.get("reason","")}, ensure_ascii=False) + "\n")
print(f"wrote -> {OUT}/contrast_{{comply,refuse}}.jsonl")
