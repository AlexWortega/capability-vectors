"""exp4 — inject wrong actions into PASS trajectories.

Take 30 PASS trajectories from rift_dataset_v2.  For each:
- find first assistant turn that emits a tool_call (or shell command)
- create paired prompts:
    original = render(messages[:k+1])         # up to + including original assistant turn
    injected = render(messages[:k] + [WRONG]) # with wrong action substituted
- write to /workspace/capvec_pf_v8_cfact/{contrast_original,contrast_injected}.jsonl
  preserving the index for downstream capture pairing.
"""
from __future__ import annotations
import json, os, random, re
from pathlib import Path
from collections import defaultdict

random.seed(43)
ROOT = "/workspace"
TRAIN = ROOT + "/rift_dataset_v2/train.jsonl"
OUT = Path(ROOT + "/capvec_pf_v8_cfact")
OUT.mkdir(exist_ok=True, parents=True)
N_TARGET = 30
MAX_CHARS = 12000

# Wrong-action templates mirror the JSON format used by the agent harness.
WRONG_ACTIONS = [
    '{"analysis": "wrong path", "command": "echo wrong > /tmp/_wrong"}',
    '{"analysis": "wrong path", "command": "ls /nonexistent_path_definitely_does_not_exist"}',
    '{"analysis": "wrong path", "search": "irrelevant unrelated query xyzzy"}',
    '<tool_call>\n{"name": "bash", "arguments": {"command": "echo wrong"}}\n</tool_call>',
]

# Detect a "decision" assistant turn: either the wrapped <tool_call> JSON or a
# bare JSON dict that carries a `command` / `search` field (claw-eval / tbench).
TOOLCALL_RE = re.compile(r"<tool_call\b", re.I)
JSON_ACTION_RE = re.compile(r'"\s*(command|search)\s*"\s*:', re.I)
CODE_FENCE_RE = re.compile(r"```\s*(?:bash|sh|shell|python)", re.I)


def find_first_action_turn(msgs):
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        ct = m.get("content") or ""
        if TOOLCALL_RE.search(ct) or JSON_ACTION_RE.search(ct) or CODE_FENCE_RE.search(ct):
            return i
    return None


def render_prefix(msgs):
    parts = [f"<|{m['role']}|>\n{m.get('content','')}" for m in msgs]
    s = "\n".join(parts)
    if len(s) > MAX_CHARS:
        s = s[-MAX_CHARS:]
    return s


def main():
    rows = [json.loads(l) for l in open(TRAIN)]
    pass_rows = [r for r in rows if r.get("reward") == 1.0]
    random.shuffle(pass_rows)
    print(f"PASS pool: {len(pass_rows)}")

    originals, injecteds = [], []
    for r in pass_rows:
        if len(originals) >= N_TARGET:
            break
        msgs = r.get("messages") or []
        k = find_first_action_turn(msgs)
        if k is None:
            continue
        # prefix = system + user + maybe prior assistant turns
        prefix = msgs[:k]
        wrong = random.choice(WRONG_ACTIONS)
        original_turn = {"role": "assistant", "content": msgs[k].get("content") or ""}
        injected_turn = {"role": "assistant", "content": wrong}

        # the prompt fed to model = prefix + decision turn (so we can grab residual at last tok)
        orig_prompt = render_prefix(prefix + [original_turn])
        inj_prompt = render_prefix(prefix + [injected_turn])

        meta = {"task_id": r.get("task_id"), "source": r.get("source"), "turn_idx": k}
        originals.append({"prompt": orig_prompt, **meta})
        injecteds.append({"prompt": inj_prompt, **meta})

    print(f"selected {len(originals)} pairs")
    with open(OUT / "contrast_comply.jsonl", "w") as f:
        # NOTE: layout matches existing capture scripts (comply = "good", refuse = "bad")
        for r in originals:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "contrast_refuse.jsonl", "w") as f:
        for r in injecteds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote -> {OUT}/contrast_{{comply,refuse}}.jsonl  (originals -> comply, injected -> refuse)")


if __name__ == "__main__":
    main()
