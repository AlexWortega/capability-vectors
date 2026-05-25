"""Build pass-vs-fail contrast for abliteration.

PRISTINE POS bucket = trajectories with reward==1.0 from rift_dataset_v2
PRISTINE NEG bucket = trajectories with reward==0.0 from same source

Each contrast entry = full chat-templated prompt+completion text.
Last-token residual at this position encodes "model just emitted PASS/FAIL trajectory".
"""
import json, os, random
random.seed(42)

SRC = "/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/rift_dataset_v2/train.jsonl"
OUT = "/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64/capvec_pf"
os.makedirs(OUT, exist_ok=True)
N_PER_BUCKET = 60   # 60 each (capture is ~3-5 sec per sample)
MAX_CHARS = 12000   # cap context to keep within model max_seq

rows = [json.loads(l) for l in open(SRC)]
pos = [r for r in rows if r["reward"] == 1.0]
neg = [r for r in rows if r["reward"] == 0.0]
print(f"pristine pos={len(pos)}  neg={len(neg)}")

def render(t):
    # render messages as a single text blob to feed model
    parts = []
    for m in t["messages"]:
        parts.append(f"<|{m['role']}|>\n{m['content']}")
    blob = "\n".join(parts)
    if len(blob) > MAX_CHARS: blob = blob[-MAX_CHARS:]  # keep tail (so last-token activation is in completion)
    return blob

random.shuffle(pos); random.shuffle(neg)
pos_pick = pos[:N_PER_BUCKET]
neg_pick = neg[:N_PER_BUCKET]
print(f"picked pos={len(pos_pick)}  neg={len(neg_pick)}")

with open(f"{OUT}/contrast_comply.jsonl", "w") as f:
    for r in pos_pick:
        f.write(json.dumps({"prompt": render(r), "source": r["source"], "task_id": r["task_id"], "reward": r["reward"]}, ensure_ascii=False) + "\n")
with open(f"{OUT}/contrast_refuse.jsonl", "w") as f:
    for r in neg_pick:
        f.write(json.dumps({"prompt": render(r), "source": r["source"], "task_id": r["task_id"], "reward": r["reward"]}, ensure_ascii=False) + "\n")
print(f"wrote -> {OUT}/contrast_{{comply,refuse}}.jsonl")
print(f"  comply (PASS) by source: {dict((s, sum(1 for r in pos_pick if r['source']==s)) for s in set(r['source'] for r in pos_pick))}")
print(f"  refuse (FAIL) by source: {dict((s, sum(1 for r in neg_pick if r['source']==s)) for s in set(r['source'] for r in neg_pick))}")
