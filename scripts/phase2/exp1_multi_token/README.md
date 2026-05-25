# exp1 — multi-token residual capture

Instead of capturing only the LAST-token residual, probe at positions
{256, 768, 1280, 1792, last}. Aggregate two ways:

- **mean** — average diff over positions, single per-layer direction
- **pca**  — concat all (N*P) centered diffs, top-1 SVD component

Pick best (layer, method) by max |AUC-0.5|. Abliterate strength=0.5, wrap, bench.

Reuses the existing PASS/FAIL bucket from `/workspace/capvec_pf_v2/` (60+60
Gemini-clean trajectories).
