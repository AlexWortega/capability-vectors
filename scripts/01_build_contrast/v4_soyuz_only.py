"""v4 contrast: only soyuz-produced trajectories (no 27b/35b/base/rift/dpo/grpo)."""
import json, os, random
random.seed(44)

ROOT = "/workspace"
CLEAN = ROOT + "/clean_fails.jsonl"
TRAIN = ROOT + "/rift_dataset_v2/train.jsonl"
OUT = ROOT + "/capvec_pf_v4_soyuzonly"
os.makedirs(OUT, exist_ok=True)
N = 60
MAX_CHARS = 12000
SOYUZ_SRC = {"tbench-2-soyuz", "claw-eval-soyuz", "mmlu-pi-soyuz-keen", "mmlu-pi-soyuz-flat"}

rows = [json.loads(l) for l in open(TRAIN)]
pos = [r for r in rows if r["source"] in SOYUZ_SRC and r["reward"] == 1.0]

# clean_fails records: {source,task_id,label,reason,messages}
clean_by_key = {}
for l in open(CLEAN):
    o = json.loads(l)
    if o["label"] == "CLEAN_FAIL":
        clean_by_key[(o["source"], o["task_id"])] = True
neg = [r for r in rows if r["source"] in SOYUZ_SRC and r["reward"] == 0.0 and (r["source"], r["task_id"]) in clean_by_key]
print(f"pos pool={len(pos)}  neg pool={len(neg)}")

def render(t):
    parts = [f"<|{m['role']}|>\n{m['content']}" for m in t["messages"]]
    s = "\n".join(parts)
    if len(s) > MAX_CHARS: s = s[-MAX_CHARS:]
    return s

random.shuffle(pos); random.shuffle(neg)
pos_pick = pos[:N]; neg_pick = neg[:N]

with open(f"{OUT}/contrast_comply.jsonl","w") as f:
    for r in pos_pick: f.write(json.dumps({"prompt": render(r), "source": r["source"], "task_id": r["task_id"]}, ensure_ascii=False)+"\n")
with open(f"{OUT}/contrast_refuse.jsonl","w") as f:
    for r in neg_pick: f.write(json.dumps({"prompt": render(r), "source": r["source"], "task_id": r["task_id"]}, ensure_ascii=False)+"\n")

from collections import Counter
print(f"picked PASS by src: {dict(Counter(r['source'] for r in pos_pick))}")
print(f"picked NEG  by src: {dict(Counter(r['source'] for r in neg_pick))}")
print(f"wrote -> {OUT}")
