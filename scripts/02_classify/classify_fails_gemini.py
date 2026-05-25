"""Use Gemini-3-flash via OpenRouter to mark CLEAN failures in our negative trajectories.

Input: rift_dataset_v2/train.jsonl, filter reward==0 negatives.
For each: send (task prompt + assistant trace tail) to Gemini, ask if the trajectory
shows a CLEAR failure pattern (model hallucinates / gives up / wrong tool / loops)
vs ambiguous (verifier issue / task underspecified).

Output: clean_fails.jsonl with {task_id, source, label: "clean_fail"|"ambiguous"|"verifier_issue", reason, original_messages}
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/home/alexw/runs/gemma4-e4b-soyuz-agenttrove-qlora-r64"
IN_FILE = ROOT + "/rift_dataset_v2/train.jsonl"
OUT_FILE = ROOT + "/clean_fails.jsonl"
OUT_AMBIG = ROOT + "/ambiguous_fails.jsonl"
KEY = open(ROOT + "/.env").read()
import re
KEY = re.search(r"OPENROUTER_API_KEY=(\S+)", KEY).group(1)
MODEL = "google/gemini-3-flash-preview"
MAX_PARALLEL = 8

PROMPT_TMPL = """You are reviewing an LLM agent trajectory that produced a FAILURE.

Classify which category this failure falls into:

A) CLEAN_FAIL — model showed a clear mistake pattern: hallucinated facts, gave up early, ran wrong tool/command, picked wrong option, repeated itself, ignored constraints, or produced obviously incorrect output. This is signal we can train on.

B) VERIFIER_ISSUE — the test/verifier might have been wrong, ambiguous, or environment-dependent. Model's behavior looks reasonable but graded fail. Noise.

C) TASK_TOO_HARD — task requires capability the model fundamentally lacks (rare specialized domain, multi-day work compressed). Not informative as training negative.

D) AMBIGUOUS — can't tell.

Trajectory (truncated for brevity):
---
{trace}
---

Respond with ONE JSON object:
{{"label":"CLEAN_FAIL"|"VERIFIER_ISSUE"|"TASK_TOO_HARD"|"AMBIGUOUS", "reason":"<one short sentence>"}}"""

def call_gemini(text, retries=3):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0.1,
        "max_tokens": 200,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    last_err = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e; time.sleep(2)
    raise last_err

def render_trace(t, max_chars=6000):
    parts = []
    for m in t["messages"][:6]:                 # head
        parts.append(f"[{m['role']}] {m['content'][:600]}")
    if len(t["messages"]) > 8:
        parts.append("... (middle omitted) ...")
        for m in t["messages"][-4:]:           # tail
            parts.append(f"[{m['role']}] {m['content'][:600]}")
    s = "\n".join(parts)
    if len(s) > max_chars: s = s[:max_chars] + "\n...(truncated)"
    return s

def classify(t):
    trace_text = render_trace(t)
    full_prompt = PROMPT_TMPL.format(trace=trace_text)
    try:
        resp = call_gemini(full_prompt)
    except Exception as e:
        return {"task_id": t["task_id"], "source": t["source"], "label": "ERROR", "reason": str(e)[:100]}
    m = re.search(r"\{.*?\}", resp, flags=re.S)
    if not m:
        return {"task_id": t["task_id"], "source": t["source"], "label": "PARSE_FAIL", "reason": resp[:200]}
    try:
        obj = json.loads(m.group(0))
        label = obj.get("label", "PARSE_FAIL")
        reason = obj.get("reason", "")
    except:
        label, reason = "PARSE_FAIL", resp[:200]
    return {"task_id": t["task_id"], "source": t["source"], "n_turns": t.get("n_turns"),
            "reward": t["reward"], "label": label, "reason": reason}

def main():
    rows = [json.loads(l) for l in open(IN_FILE)]
    negs = [r for r in rows if r["reward"] == 0]
    print(f"negatives total: {len(negs)}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = {ex.submit(classify, t): t for t in negs}
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            results.append(r)
            if (i+1) % 20 == 0:
                from collections import Counter
                c = Counter(r["label"] for r in results)
                print(f"  {i+1}/{len(negs)}  dist={dict(c)}", flush=True)

    from collections import Counter
    print("\nFinal label distribution:")
    for label, n in Counter(r["label"] for r in results).most_common():
        print(f"  {label:<20} {n}")

    # Save splits
    clean = []
    ambig = []
    task_to_msgs = {(r["source"], r["task_id"]): r["messages"] for r in negs for src,tid in [(r["source"], r["task_id"])]}
    src_id_to_orig = {(t["source"], t["task_id"]): t for t in negs}
    for r in results:
        key = (r["source"], r["task_id"])
        orig = src_id_to_orig.get(key, {})
        out = {**r, "messages": orig.get("messages", [])}
        if r["label"] == "CLEAN_FAIL": clean.append(out)
        else: ambig.append(out)
    with open(OUT_FILE, "w") as f:
        for r in clean: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_AMBIG, "w") as f:
        for r in ambig: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote CLEAN_FAIL {len(clean)} → {OUT_FILE}")
    print(f"wrote others    {len(ambig)} → {OUT_AMBIG}")

if __name__ == "__main__":
    main()
