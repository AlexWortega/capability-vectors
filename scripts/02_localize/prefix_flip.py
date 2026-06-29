"""First-error localization via binary search (Who & When — arxiv:2505.00212).

For each FAIL trace: binary search for the first turn such that if we truncate the
conversation at that turn and regenerate the suffix, the outcome flips to PASS.
That turn index is `first_error_step`.

Usage:
    python scripts/02_localize/prefix_flip.py \
        --rollouts /workspace/trajedit/rollouts_qwen3_8b_ha20.jsonl \
        --out /workspace/trajedit/rollouts_localized.jsonl \
        --base-url http://localhost:30040/v1 \
        --model qwen3_8b

Output: same JSONL with added field `first_error_step: int | null`.
  null = localization failed or trace too short (≤2 assistant turns).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Any
from openai import OpenAI

MAX_TURNS = 20
MAX_TOKENS = 4096
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them step by step to complete the task. "
    "Call tools exactly as specified — do not hallucinate tool names."
)


def simulate_tool(tool_name: str, args: dict) -> str:
    return f"[tool_result: {tool_name} called with args={json.dumps(args)[:200]}]"


def rebuild_openai_messages(turns: list[dict]) -> list[dict]:
    """Convert trace turns back to OpenAI message format."""
    messages = []
    for t in turns:
        role = t["role"]
        if role == "system":
            messages.append({"role": "system", "content": t["content"]})
        elif role == "user":
            messages.append({"role": "user", "content": t["content"]})
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": t.get("content") or ""}
            if t.get("tool_calls"):
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": tc["function"],
                    }
                    for tc in t["tool_calls"]
                ]
            messages.append(msg)
        elif role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": t["tool_call_id"],
                "content": t["content"],
            })
    return messages


def regenerate_suffix(
    client: OpenAI,
    model: str,
    prefix_turns: list[dict],
    scenario_instruction: str,
    tools: list[dict],
) -> list[dict]:
    """Regenerate suffix from prefix_turns, return completed turns."""
    # Build messages: system + user instruction + prefix turns
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario_instruction},
    ]
    messages += rebuild_openai_messages(prefix_turns)

    new_turns = []
    for _ in range(MAX_TURNS - len(prefix_turns)):
        kwargs = dict(model=model, messages=messages, max_tokens=MAX_TOKENS, temperature=0.0)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        turn = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "function": {"name": tc.function.name,
                                           "arguments": tc.function.arguments}}
                for tc in (msg.tool_calls or [])
            ],
        }
        new_turns.append(turn)
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            fn_args = {}
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                pass
            result = simulate_tool(tc.function.name, fn_args)
            new_turns.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return new_turns


def heuristic_outcome(turns: list[dict]) -> str:
    """Simple heuristic: PASS if last assistant message is non-empty and non-error."""
    last_asst = next((t for t in reversed(turns) if t["role"] == "assistant"), None)
    if not last_asst or not last_asst.get("content"):
        return "fail"
    content = last_asst["content"].lower()
    if any(kw in content for kw in ("error", "failed", "cannot", "unable")):
        return "fail"
    return "pass"


def localize_first_error(
    client: OpenAI,
    model: str,
    fail_trace: dict,
    verifier_fn=None,
) -> int | None:
    """Binary search: return index of first decisive error step (0-indexed assistant turns).

    Returns None if trace has ≤2 assistant turns or if the flip is never found.
    """
    asst_turns = [i for i, t in enumerate(fail_trace["turns"]) if t["role"] == "assistant"]
    if len(asst_turns) <= 2:
        return None

    instruction = fail_trace.get("instruction", "")
    tools = fail_trace.get("tools", [])

    def check_flip(prefix_len: int) -> bool:
        """True if truncating to prefix_len assistant turns → outcome flips to PASS."""
        prefix_asst_idxs = asst_turns[:prefix_len]
        # Include all turns up to (and including) the prefix_len-th assistant turn
        if not prefix_asst_idxs:
            cutoff = 0
        else:
            cutoff = prefix_asst_idxs[-1] + 1
        # Also include subsequent tool-result turns from that last assistant call
        # (the model needs to see the tool result from the last action in the prefix)
        while cutoff < len(fail_trace["turns"]) and fail_trace["turns"][cutoff]["role"] == "tool":
            cutoff += 1
        prefix_turns = fail_trace["turns"][:cutoff]

        suffix_turns = regenerate_suffix(client, model, prefix_turns, instruction, tools)
        all_turns = prefix_turns + suffix_turns

        if verifier_fn is not None:
            return verifier_fn(all_turns)
        return heuristic_outcome(all_turns) == "pass"

    lo, hi = 1, len(asst_turns)
    found = None
    while lo < hi:
        mid = (lo + hi) // 2
        flip = check_flip(mid)
        if flip:
            # Flipped to PASS → error is at or after this prefix → search earlier
            hi = mid
            found = mid
        else:
            # Still FAIL → error is before this prefix → search later
            lo = mid + 1

    # found = smallest prefix_len s.t. regenerating suffix gives PASS
    # first_error_step = asst_turns[found - 1] (0-indexed into turns list)
    if found is None:
        return None
    return asst_turns[found - 1] if found > 0 else asst_turns[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:30040/v1")
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--max-traces", type=int, default=200,
                    help="Max FAIL traces to localize")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    traces = [json.loads(l) for l in open(args.rollouts)]
    fail_traces = [t for t in traces if t["outcome"] == "fail"][:args.max_traces]
    pass_traces = [t for t in traces if t["outcome"] == "pass"]
    print(f"[init] fail={len(fail_traces)} pass={len(pass_traces)}", flush=True)

    localized = 0
    with open(out_path, "w") as fout:
        # Write PASS traces unchanged
        for t in pass_traces:
            t["first_error_step"] = None
            fout.write(json.dumps(t) + "\n")

        # Localize FAIL traces
        for i, trace in enumerate(fail_traces):
            t0 = time.time()
            try:
                step = localize_first_error(client, args.model, trace)
            except Exception as e:
                print(f"  [warn] {trace['task_id']} localization failed: {e}", flush=True)
                step = None

            trace["first_error_step"] = step
            fout.write(json.dumps(trace) + "\n")
            fout.flush()

            if step is not None:
                localized += 1
            dur = time.time() - t0
            print(
                f"[{i+1}/{len(fail_traces)}] {trace['task_id']:30s} "
                f"first_error_step={step}  dur={dur:.1f}s  localized={localized}",
                flush=True,
            )

    print(f"[done] localized {localized}/{len(fail_traces)} FAIL traces  → {out_path}", flush=True)


if __name__ == "__main__":
    main()
