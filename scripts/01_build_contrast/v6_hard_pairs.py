"""v6 hard pairs (expanded): same task_id from any source.
For tbench tasks: same prompt across soyuz/base/rift/dpo/grpo/27b/35b — already same env.
For mmlu-pi: same question_id across runs.
For claw-eval: each task appears once (no pairs from there).
"""
import json, os
from collections import defaultdict

ROOT = "/workspace"
OUT = ROOT + "/capvec_pf_v6_hardpairs"
os.makedirs(OUT, exist_ok=True)

rows = [json.loads(l) for l in open(f"{ROOT}/rift_dataset_v2/train.jsonl")]
by_tid = defaultdict(lambda: {"pass": [], "fail": []})
for r in rows:
    tid = r["task_id"]
    if r["reward"] == 1.0: by_tid[tid]["pass"].append(r)
    elif r["reward"] == 0.0: by_tid[tid]["fail"].append(r)

hard = {tid: v for tid, v in by_tid.items() if v["pass"] and v["fail"]}
print(f"all rows: {len(rows)}  unique task_ids: {len(by_tid)}  hard-pair task_ids: {len(hard)}", flush=True)

def render(t, MAX=12000):
    parts = [f"<|{m['role']}|>\n{m['content']}" for m in t["messages"]]
    s = "\n".join(parts)
    return s[-MAX:] if len(s) > MAX else s

pos_rows = []; neg_rows = []
for tid, v in hard.items():
    pos_rows.append({"prompt": render(v["pass"][0]), "task_id": tid, "source": v["pass"][0]["source"]})
    neg_rows.append({"prompt": render(v["fail"][0]), "task_id": tid, "source": v["fail"][0]["source"]})

print(f"pos={len(pos_rows)} neg={len(neg_rows)}")
with open(f"{OUT}/contrast_comply.jsonl","w") as f:
    for r in pos_rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
with open(f"{OUT}/contrast_refuse.jsonl","w") as f:
    for r in neg_rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
print(f"wrote -> {OUT}")
from collections import Counter
print("PASS by src:", Counter(r["source"] for r in pos_rows))
print("FAIL by src:", Counter(r["source"] for r in neg_rows))
