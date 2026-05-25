# exp2 — within-task hard pairs

Reuses `/workspace/capvec_pf_v6_hardpairs/contrast_{comply,refuse}.jsonl`
built by `scripts/01_build_contrast/v6_hard_pairs.py` (51 unique task_ids
with both a PASS rollout and a FAIL rollout — same prompt, different models).

Within-task contrast removes task-difficulty confound, should yield a
cleaner direction than v2 (60 random PASS vs 60 random FAIL).

`run.sh` chains capture → compute_dir → abliterate → wrap.
