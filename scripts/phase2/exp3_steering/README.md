# exp3 — activation steering at inference

Custom FastAPI OpenAI-compat server (`steered_openai_server.py`) loads
soyuz-merged via transformers, registers a `forward_pre_hook` on
`model.model.layers[L].input_layernorm` that adds `alpha * direction` to the
residual stream. Direction is loaded from any `pf_dir_L*.pt`.

The server exposes a runtime `/steering` POST so we can sweep
`(alpha, layer)` without reloading the model. `sweep_alpha.sh` walks
alpha ∈ {-2,-1,-0.5,0,0.5,1,2} × layer ∈ {10,16,27} and runs HA20 for each.

tbench-17 is run only on the (alpha, layer) winner of HA20 (saves ~12 hours
of compute given how slow transformers single-stream generation is vs sglang).

Postponed to last because (a) it requires a custom server, (b) generation is
slower than sglang batched. Skippable if 1-4 already deliver a winner.
