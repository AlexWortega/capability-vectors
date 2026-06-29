"""Capture residuals at localized steps for TrajEdit direction computation.

Unlike capture_pf.py (which takes last-token of the full sequence), this script
captures the residual at the SPECIFIC TURN POSITION indicated by:
  - For FAIL traces: `first_error_step` (index in turns list)
  - For PASS traces: `recovery_step` (index in turns list)

Fallback: if step is None, falls back to last-token (same as capture_pf.py) and
sets `capture_mode="outcome_fallback"` in the output metadata.

Also captures outcome-position activations (for ablation comparison in
compute_trajedit_dir.py).

Output layout:
    /workspace/trajedit/activations/
        fail_firsterror.npy   [N_fail, L+1, D]
        pass_recovery.npy     [N_pass, L+1, D]
        fail_outcome.npy      [N_fail, L+1, D]  ← same as capture_pf.py
        pass_outcome.npy      [N_pass, L+1, D]

Usage:
    python scripts/03_capture/capture_trajedit.py \
        --rollouts /workspace/trajedit/rollouts_with_recovery.jsonl \
        --model-path Qwen/Qwen3-8B-Instruct \
        --out-dir /workspace/trajedit/activations
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def turns_to_text(turns: list[dict], tokenizer) -> str:
    """Render turns as a chat-templated string (same as contrast files)."""
    messages = []
    for t in turns:
        role = t["role"]
        if role in ("system", "user", "assistant"):
            messages.append({"role": role, "content": t.get("content") or ""})
        elif role == "tool":
            # Render tool results as user messages for models without native tool role
            messages.append({"role": "user", "content": f"[tool result]: {t.get('content','')}"})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def find_turn_token_position(
    input_ids: torch.Tensor,
    tokenizer,
    turns: list[dict],
    target_turn_idx: int,
) -> int:
    """Return token position of the last token of the turn at target_turn_idx.

    Strategy: tokenize prefix turns, take len(prefix_ids) - 1 as position.
    If the position is out of range, clamp to last token.
    """
    prefix_turns = turns[: target_turn_idx + 1]
    prefix_text = turns_to_text(prefix_turns, tokenizer)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    pos = min(len(prefix_ids) - 1, input_ids.shape[1] - 1)
    return max(pos, 0)


@torch.no_grad()
def capture_one(
    model,
    tokenizer,
    turns: list[dict],
    localized_step: int | None,
    max_tokens: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (acts_at_step, acts_at_outcome), each [L+1, D]."""
    full_text = turns_to_text(turns, tokenizer)
    ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] > max_tokens:
        ids = ids[:, -max_tokens:]
    ids = ids.to(model.device)

    out = model(input_ids=ids, output_hidden_states=True, return_dict=True, use_cache=False)
    hidden = out.hidden_states  # tuple of [1, T, D], length L+1

    # Outcome position: last token of full sequence
    outcome_pos = ids.shape[1] - 1
    acts_outcome = np.stack(
        [h[0, outcome_pos, :].float().cpu().numpy() for h in hidden], axis=0
    )  # [L+1, D]

    # Localized step position
    if localized_step is not None:
        step_pos = find_turn_token_position(ids, tokenizer, turns, localized_step)
    else:
        step_pos = outcome_pos  # fallback

    acts_step = np.stack(
        [h[0, step_pos, :].float().cpu().numpy() for h in hidden], axis=0
    )  # [L+1, D]

    return acts_step, acts_outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen3-8B-Instruct")
    ap.add_argument("--out-dir", default="/workspace/trajedit/activations")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = [json.loads(l) for l in open(args.rollouts)]
    fail_traces = [t for t in traces if t["outcome"] == "fail"]
    pass_traces = [t for t in traces if t["outcome"] == "pass"]
    print(f"[init] fail={len(fail_traces)} pass={len(pass_traces)}", flush=True)

    print(f"[load] {args.model_path}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.eval()
    nl = model.config.num_hidden_layers
    dm = model.config.hidden_size
    print(f"[load] n_layers={nl} d_model={dm} took={time.time()-t0:.1f}s", flush=True)

    # Capture FAIL traces
    fail_step_acts, fail_outcome_acts = [], []
    fail_meta = []
    print("[capture FAIL traces]", flush=True)
    for i, trace in enumerate(fail_traces):
        step_idx = trace.get("first_error_step")
        acts_step, acts_outcome = capture_one(
            model, tok, trace["turns"], step_idx, args.max_tokens
        )
        fail_step_acts.append(acts_step)
        fail_outcome_acts.append(acts_outcome)
        fail_meta.append({
            "task_id": trace["task_id"],
            "first_error_step": step_idx,
            "capture_mode": "first_error" if step_idx is not None else "outcome_fallback",
        })
        if (i + 1) % 10 == 0:
            print(f"  FAIL {i+1}/{len(fail_traces)}", flush=True)

    # Capture PASS traces
    pass_step_acts, pass_outcome_acts = [], []
    pass_meta = []
    print("[capture PASS traces]", flush=True)
    for i, trace in enumerate(pass_traces):
        step_idx = trace.get("recovery_step")
        acts_step, acts_outcome = capture_one(
            model, tok, trace["turns"], step_idx, args.max_tokens
        )
        pass_step_acts.append(acts_step)
        pass_outcome_acts.append(acts_outcome)
        pass_meta.append({
            "task_id": trace["task_id"],
            "recovery_step": step_idx,
            "capture_mode": "recovery" if step_idx is not None else "outcome_fallback",
        })
        if (i + 1) % 10 == 0:
            print(f"  PASS {i+1}/{len(pass_traces)}", flush=True)

    # Stack and save
    fail_step_arr = np.stack(fail_step_acts, axis=0)     # [N_fail, L+1, D]
    fail_out_arr = np.stack(fail_outcome_acts, axis=0)
    pass_step_arr = np.stack(pass_step_acts, axis=0)
    pass_out_arr = np.stack(pass_outcome_acts, axis=0)

    print(f"fail_firsterror={fail_step_arr.shape}  pass_recovery={pass_step_arr.shape}", flush=True)
    np.save(out_dir / "fail_firsterror.npy", fail_step_arr)
    np.save(out_dir / "pass_recovery.npy", pass_step_arr)
    np.save(out_dir / "fail_outcome.npy", fail_out_arr)
    np.save(out_dir / "pass_outcome.npy", pass_out_arr)

    import json as _json
    (out_dir / "fail_meta.json").write_text(_json.dumps(fail_meta, indent=2))
    (out_dir / "pass_meta.json").write_text(_json.dumps(pass_meta, indent=2))

    print(f"[done] saved to {out_dir}", flush=True)
    print(f"  fail_firsterror.npy  {fail_step_arr.shape}", flush=True)
    print(f"  pass_recovery.npy    {pass_step_arr.shape}", flush=True)
    print(f"  fail_outcome.npy     {fail_out_arr.shape}", flush=True)
    print(f"  pass_outcome.npy     {pass_out_arr.shape}", flush=True)


if __name__ == "__main__":
    main()
