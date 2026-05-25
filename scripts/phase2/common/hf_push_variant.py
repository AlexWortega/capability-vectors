"""Push a single phase2 abliterated VLM to HuggingFace.

Usage:
  HF_TOKEN=... python hf_push_variant.py \
      --tag exp5_agentonly \
      --local /workspace/abliterated_soyuz_v7_agentonly_vlm \
      --tbench 3/17 --ha20 9 --mmlu 47.2 --method "agent-only mean L=14, s=0.5" \
      --notes "drop mmlu-pi from FAIL bucket; tests MMLU recovery hypothesis"
"""
from __future__ import annotations
import argparse, os
from huggingface_hub import create_repo, upload_folder


CARD_TEMPLATE = """---
library_name: transformers
base_model: Qwen/Qwen3.5-4B
license: apache-2.0
language: [en]
tags:
- agent
- abliteration
- orthogonalization
- weight-ortho
- soyuz
- qwen3.5
- phase2
datasets:
- AlexWortega/Soyuz-sft
---

# Qwen3.5-4B Soyuz — Abliterated ({tag})

Phase-2 weight-orthogonalized variant of
[`AlexWortega/qwen35-4b-soyuz-merged`](https://huggingface.co/AlexWortega/qwen35-4b-soyuz-merged).

| field | value |
| --- | --- |
| **Method** | {method} |
| **tbench-2 (17)** | **{tbench}** |
| **HermesAgent-20** | **{ha20} / 20** |
| **MMLU-Pro** | {mmlu} |
| **EQbench3** | {eq} |
| Notes | {notes} |

## Lineage

Continues the [`capability-vectors`](https://github.com/AlexWortega/capability-vectors)
sweep. Phase 1 best was `v2` (HA20 8/20, MMLU collapse 58→2). Phase 2 explores
multi-token / hard-pairs / counterfactual / agent-only / activation-steering recipes.

See https://github.com/AlexWortega/capability-vectors for repo + per-experiment
README, and `phase2/results/all_variants.csv` for the live results table.

## Usage with sglang

```bash
python -m sglang.launch_server \\
    --model-path AlexWortega/{repo_name} \\
    --dtype bfloat16 --trust-remote-code \\
    --tool-call-parser hermes --chat-template hermes_qwen.jinja
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--repo", default=None,
                    help="default AlexWortega/qwen35-4b-soyuz-abliterated-<tag>")
    ap.add_argument("--method", default="")
    ap.add_argument("--tbench", default="—")
    ap.add_argument("--ha20", default="—")
    ap.add_argument("--mmlu", default="—")
    ap.add_argument("--eq", default="—")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN env var required")

    repo = args.repo or f"AlexWortega/qwen35-4b-soyuz-abliterated-{args.tag}"
    repo_name = repo.split("/")[-1]
    create_repo(repo, exist_ok=True, repo_type="model", token=token)

    card = CARD_TEMPLATE.format(
        tag=args.tag, method=args.method, tbench=args.tbench,
        ha20=args.ha20, mmlu=args.mmlu, eq=args.eq, notes=args.notes,
        repo_name=repo_name,
    )
    readme = os.path.join(args.local, "README.md")
    with open(readme, "w") as f:
        f.write(card)
    print(f"[push] {args.local} -> {repo}", flush=True)
    upload_folder(repo_id=repo, folder_path=args.local, token=token,
                  commit_message=f"phase2 abliterated {args.tag}")
    print(f"done -> https://huggingface.co/{repo}", flush=True)


if __name__ == "__main__":
    main()
