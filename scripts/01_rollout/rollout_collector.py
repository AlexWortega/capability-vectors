"""Collect PASS/FAIL rollout traces from Qwen3-8B on HA20-style scenarios.

Usage:
    # Serve first:
    python -m sglang.launch_server --model-path Qwen/Qwen3-8B-Instruct \
        --served-model-name qwen3_8b --port 30040 --tool-call-parser hermes \
        --chat-template hermes_qwen.jinja --trust-remote-code

    # Then collect:
    python scripts/01_rollout/rollout_collector.py \
        --scenarios /tmp/HermesAgent-20/scenarios \
        --out /workspace/trajedit/rollouts_qwen3_8b_ha20.jsonl \
        --base-url http://localhost:30040/v1 \
        --model qwen3_8b \
        --n-repeats 3
"""
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path
from typing import Any
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them step by step to complete the task. "
    "Call tools exactly as specified — do not hallucinate tool names."
)

MAX_TURNS = 20
MAX_TOKENS = 4096


def load_scenarios(scenarios_dir: Path) -> list[dict]:
    """Load HA20 scenario JSONs: {id, instruction, tools, verifier}."""
    scenarios = []
    for p in sorted(scenarios_dir.glob("*.json")):
        with open(p) as f:
            s = json.load(f)
        s.setdefault("id", p.stem)
        scenarios.append(s)
    return scenarios


def simulate_tool(tool_name: str, args: dict) -> str:
    """Stub — replace with real tool dispatch in production."""
    return f"[tool_result: {tool_name} called with args={json.dumps(args)[:200]}]"


def run_scenario(
    client: OpenAI,
    model: str,
    scenario: dict,
    verifier_fn: Any = None,
) -> dict:
    """Run one scenario; return trace dict with outcome."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario["instruction"]},
    ]
    tools = scenario.get("tools", [])
    turns: list[dict] = []
    outcome = "fail"

    for turn_idx in range(MAX_TURNS):
        kwargs = dict(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            turns.append({"role": "error", "content": str(e), "turn": turn_idx})
            break

        msg = resp.choices[0].message
        turns.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (msg.tool_calls or [])
            ],
            "turn": turn_idx,
        })
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # Model finished without more tool calls
            break

        # Execute tool calls
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            result = simulate_tool(fn_name, fn_args)
            turns.append({"role": "tool", "tool_call_id": tc.id,
                          "content": result, "turn": turn_idx})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Determine outcome
    if verifier_fn is not None:
        try:
            outcome = "pass" if verifier_fn(turns) else "fail"
        except Exception:
            outcome = "fail"
    else:
        # Heuristic: last assistant message has no tool calls and content is non-empty
        last_asst = next(
            (t for t in reversed(turns) if t["role"] == "assistant"), None
        )
        if last_asst and last_asst.get("content") and not last_asst.get("tool_calls"):
            outcome = "pass" if "error" not in last_asst["content"].lower() else "fail"

    return {
        "task_id": scenario["id"],
        "source": "ha20",
        "turns": turns,
        "n_turns": len([t for t in turns if t["role"] == "assistant"]),
        "outcome": outcome,
        "reward": 1.0 if outcome == "pass" else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="/tmp/HermesAgent-20/scenarios",
                    help="Dir of scenario JSON files")
    ap.add_argument("--out", default="/workspace/trajedit/rollouts_qwen3_8b_ha20.jsonl")
    ap.add_argument("--base-url", default="http://localhost:30040/v1")
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--n-repeats", type=int, default=3,
                    help="Runs per scenario to get diverse pass/fail examples")
    ap.add_argument("--target-fail", type=int, default=200,
                    help="Stop after collecting this many FAIL traces")
    ap.add_argument("--target-pass", type=int, default=100)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key="dummy")
    scenarios = load_scenarios(Path(args.scenarios))
    print(f"[init] {len(scenarios)} scenarios  repeats={args.n_repeats}  "
          f"targets: fail≥{args.target_fail} pass≥{args.target_pass}", flush=True)

    n_pass = n_fail = 0
    with open(out_path, "w") as fout:
        for repeat in range(args.n_repeats):
            for sc in scenarios:
                if n_pass >= args.target_pass and n_fail >= args.target_fail:
                    print(f"[done] targets reached: pass={n_pass} fail={n_fail}", flush=True)
                    return

                t0 = time.time()
                try:
                    trace = run_scenario(client, args.model, sc)
                except Exception:
                    traceback.print_exc()
                    continue

                trace["repeat"] = repeat
                fout.write(json.dumps(trace) + "\n")
                fout.flush()

                if trace["outcome"] == "pass":
                    n_pass += 1
                else:
                    n_fail += 1

                dur = time.time() - t0
                print(
                    f"[rep={repeat}] {sc['id']:30s} outcome={trace['outcome']} "
                    f"turns={trace['n_turns']} dur={dur:.1f}s | "
                    f"total pass={n_pass} fail={n_fail}",
                    flush=True,
                )

    print(f"[done] saved {n_pass+n_fail} traces to {out_path}  "
          f"(pass={n_pass} fail={n_fail})", flush=True)


if __name__ == "__main__":
    main()
