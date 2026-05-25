"""exp3 — transformers + FastAPI OpenAI-compat server with residual steering.

Loads the SOYUZ-merged text-only model on GPU and registers a
`forward_pre_hook` on `model.model.layers[L].input_layernorm` that adds
`alpha * direction` to the residual stream input. Exposes the chat
completion endpoint compatible with sglang / openai-python.

POST /v1/chat/completions    (subset of OpenAI; tool_calls parsed from
                              <tool_call>{json}</tool_call> blocks)
GET  /v1/models
GET  /steering                (current alpha, layer, direction file)
POST /steering                {"alpha": 0.5, "layer": 16, "dir_pt": "..."}

Env knobs:
  STEER_MODEL_PATH=/workspace/ckpts_qwen35_soyuz/clawd_soyuz_merged
  STEER_DIR_PT=/workspace/capvec_pf_v7_agentonly/vectors/pf_dir_L16.pt
  STEER_ALPHA=0.0
  STEER_LAYER=16
  STEER_SERVED_NAME=soyuz_steered
  STEER_PORT=30090
"""
from __future__ import annotations
import json, os, re, time, uuid, threading
from typing import Optional, Any
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = os.environ.get("STEER_MODEL_PATH", "/workspace/ckpts_qwen35_soyuz/clawd_soyuz_merged")
DIR_PT = os.environ.get("STEER_DIR_PT", "")
ALPHA = float(os.environ.get("STEER_ALPHA", "0.0"))
LAYER = int(os.environ.get("STEER_LAYER", "16"))
SERVED_NAME = os.environ.get("STEER_SERVED_NAME", "soyuz_steered")
PORT = int(os.environ.get("STEER_PORT", "30090"))


class SteerState:
    def __init__(self):
        self.alpha = ALPHA
        self.layer = LAYER
        self.dir_pt = DIR_PT
        self.direction: Optional[torch.Tensor] = None
        self.dtype = torch.bfloat16
        self.device = "cuda:0"
        self.lock = threading.Lock()
        self.hook_handle = None

    def load_direction(self, path: str):
        blob = torch.load(path, weights_only=False, map_location="cpu")
        d = blob["dir"].float()
        d = d / (d.norm() + 1e-9)
        self.direction = d.to(device=self.device, dtype=self.dtype)
        if "layer" in blob:
            self.layer = int(blob["layer"])
        self.dir_pt = path
        print(f"[steer] loaded direction L={self.layer} from {path}", flush=True)

    def install_hook(self, model):
        # Remove old hook if any
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        if self.direction is None or self.alpha == 0.0:
            return
        layers = model.model.layers
        if self.layer < 0 or self.layer >= len(layers):
            print(f"[steer] layer {self.layer} out of range 0..{len(layers)-1}", flush=True)
            return
        target = layers[self.layer].input_layernorm
        d = self.direction
        a = self.alpha

        def pre_hook(_module, inputs):
            # inputs[0] is the residual stream tensor [B, T, D]
            x = inputs[0]
            x = x + a * d.to(x.dtype)
            return (x,) + tuple(inputs[1:])

        self.hook_handle = target.register_forward_pre_hook(pre_hook)
        print(f"[steer] installed pre-hook on layers[{self.layer}].input_layernorm alpha={a}", flush=True)


STATE = SteerState()
TOK = None
MODEL = None


def init_model():
    global TOK, MODEL
    print(f"[load] {MODEL_PATH}", flush=True); t0 = time.time()
    TOK = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    MODEL.eval()
    print(f"[load] took={time.time()-t0:.1f}s n_layers={MODEL.config.num_hidden_layers}", flush=True)
    if STATE.dir_pt:
        STATE.load_direction(STATE.dir_pt)
    STATE.install_hook(MODEL)


app = FastAPI()
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.S)


def _emit_call(name, arguments):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def parse_tool_calls(text: str):
    """Recognise both the hermes <tool_call>...</tool_call> wrapper and the
    bare JSON form ({"analysis": "...", "command": "..."} or
    {"analysis": "...", "search": "..."}) used by claw-eval/tbench rollouts."""
    calls = []
    # 1) explicit hermes wrapper
    for m in TOOL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name") or obj.get("tool") or "tool"
            args = obj.get("arguments") if "arguments" in obj else obj
            calls.append(_emit_call(name, args))
        except Exception:
            continue
    if calls:
        return calls

    # 2) bare-JSON form with command/search field
    for m in JSON_OBJ_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "command" in obj:
            calls.append(_emit_call("bash", {"command": obj["command"]}))
        elif "search" in obj:
            calls.append(_emit_call("search", {"query": obj["search"],
                                                "limit": obj.get("limit", 5)}))
    return calls


def render_messages(messages, tools=None):
    """Render an OpenAI-style messages list using the model's chat template."""
    return TOK.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools,
    )


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": SERVED_NAME, "object": "model", "owned_by": "phase2"}
    ]}


@app.get("/steering")
def get_steer():
    return {"alpha": STATE.alpha, "layer": STATE.layer, "dir_pt": STATE.dir_pt,
            "loaded": STATE.direction is not None}


@app.post("/steering")
async def set_steer(req: Request):
    body = await req.json()
    with STATE.lock:
        if "dir_pt" in body and body["dir_pt"] and body["dir_pt"] != STATE.dir_pt:
            STATE.load_direction(body["dir_pt"])
        if "alpha" in body:
            STATE.alpha = float(body["alpha"])
        if "layer" in body:
            STATE.layer = int(body["layer"])
        STATE.install_hook(MODEL)
    return {"alpha": STATE.alpha, "layer": STATE.layer, "dir_pt": STATE.dir_pt}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    messages = body["messages"]
    tools = body.get("tools")
    max_tokens = int(body.get("max_tokens", 1024))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.95))
    stream = bool(body.get("stream", False))

    try:
        prompt = render_messages(messages, tools=tools)
    except Exception:
        prompt = "\n".join(f"{m.get('role')}: {m.get('content','')}" for m in messages)

    inputs = TOK(prompt, return_tensors="pt", add_special_tokens=False).to(MODEL.device)
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=TOK.eos_token_id,
        eos_token_id=TOK.eos_token_id,
    )

    with torch.no_grad():
        out = MODEL.generate(**inputs, **gen_kwargs)
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = TOK.decode(gen_ids, skip_special_tokens=False)
    # Cut on eos / im_end
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if stop in text:
            text = text.split(stop)[0]
    content_clean = re.sub(TOOL_RE, "", text).strip()
    tool_calls = parse_tool_calls(text)

    msg: dict[str, Any] = {"role": "assistant", "content": content_clean}
    if tool_calls:
        msg["tool_calls"] = tool_calls

    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SERVED_NAME,
        "choices": [{
            "index": 0,
            "message": msg,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": {
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(gen_ids.shape[0]),
            "total_tokens": int(inputs["input_ids"].shape[1] + gen_ids.shape[0]),
        },
    }
    if stream:
        # Minimal SSE: send the whole thing as one chunk + [DONE].
        async def gen():
            yield f"data: {json.dumps(resp)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(resp)


if __name__ == "__main__":
    init_model()
    print(f"[serve] :{PORT}  served-name={SERVED_NAME}  alpha={STATE.alpha}  L={STATE.layer}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
