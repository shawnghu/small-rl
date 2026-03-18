"""HTTP server wrapping the synchronous vLLM engine with MLP adapter support.

Drop-in replacement for vllm_server.py (ZMQ) using HTTP/msgpack transport.
Uses the sync LLM engine from vllm_mlp_adapter.py.

Usage:
    CUDA_VISIBLE_DEVICES=1 .venv-vllm/bin/python vllm_http_server.py \
        --model SimpleStories/SimpleStories-1.25M --max_experiments 10 \
        --mlp_config m16 --port 8100

The server exposes these endpoints:
    POST /register          -> {"experiment_id": int}
    POST /update_weights    -> {"ok": true}
    POST /generate          -> {"completion_texts": [...], "completion_ids": [...], ...}
    POST /set_scales        -> {"ok": true}
    POST /shutdown          -> {"ok": true}
    GET  /health            -> {"status": "ok", "registered": int, "max_experiments": int}

All POST endpoints accept and return msgpack-encoded bodies.
"""

import argparse
import os
import signal
import threading
import time

import msgpack
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response

from vllm_grpo import MLP_PRESETS, flatten_vllm_outputs
from vllm_mlp_adapter import create_engine

# Weight tensor names in order (must match client)
WEIGHT_KEYS = [
    "gate_retain", "up_retain", "down_retain",
    "gate_forget", "up_forget", "down_forget",
]


def _msgpack_response(data, status=200):
    return Response(
        content=msgpack.packb(data, use_bin_type=True),
        status_code=status,
        media_type="application/x-msgpack",
    )


def create_app(model_name, max_experiments, retain_neurons, forget_neurons,
               gpu_memory_utilization=0.05, dtype="bfloat16"):
    """Create FastAPI app wrapping a sync vLLM engine with MLP adapters."""

    app = FastAPI()

    # Engine state
    state = {
        "llm": None,
        "mgr": None,
        "next_experiment_id": 1,
        "n_registered": 0,
        "lock": threading.Lock(),
    }

    # Create engine eagerly
    print(f"[HTTPServer] Creating vLLM engine (max_experiments={max_experiments}, "
          f"retain={retain_neurons}, forget={forget_neurons}, model={model_name})...")
    t0 = time.time()
    llm, mgr = create_engine(
        model_name=model_name,
        max_experiments=max_experiments,
        retain_neurons=retain_neurons,
        forget_neurons=forget_neurons,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    state["llm"] = llm
    state["mgr"] = mgr
    print(f"[HTTPServer] Engine ready in {time.time() - t0:.1f}s")

    @app.get("/health")
    def health():
        return _msgpack_response({
            "status": "ok",
            "registered": state["n_registered"],
            "max_experiments": max_experiments,
        })

    @app.post("/register")
    def register():
        with state["lock"]:
            eid = state["next_experiment_id"]
            assert eid <= max_experiments, \
                f"Cannot register: {eid} > max_experiments={max_experiments}"
            state["next_experiment_id"] += 1
            state["n_registered"] += 1
        print(f"[HTTPServer] Registered experiment {eid} ({state['n_registered']} total)")
        return _msgpack_response({"experiment_id": eid})

    @app.post("/update_weights")
    async def update_weights(request: Request):
        body = await request.body()
        msg = msgpack.unpackb(body, raw=False)
        eid = msg["experiment_id"]
        dtype_str = msg["dtype"]
        np_dtype = np.float32 if dtype_str == "float32" else np.float16
        torch_dtype = torch.float32 if dtype_str == "float32" else torch.float16

        layer_weights = []
        for layer_data in msg["layers"]:
            w = {}
            for key in WEIGHT_KEYS:
                raw = layer_data.get(key)
                if raw is not None:
                    shape = tuple(msg["shapes"][key])
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                    w[key] = torch.from_numpy(arr.copy())
                    assert w[key].dtype == torch_dtype
            layer_weights.append(w)

        state["mgr"].set_weights(eid, layer_weights)
        return _msgpack_response({"ok": True})

    @app.post("/generate")
    async def generate(request: Request):
        from vllm import SamplingParams

        body = await request.body()
        msg = msgpack.unpackb(body, raw=False)
        eid = msg["experiment_id"]
        prompt_ids = msg["prompt_ids"]
        sp = SamplingParams(
            n=msg["n"],
            temperature=msg["temperature"],
            max_tokens=msg["max_tokens"],
        )

        t0 = time.time()
        outputs = state["mgr"].generate(prompt_ids, [eid] * len(prompt_ids), sp)
        comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(outputs)
        gen_ms = (time.time() - t0) * 1000
        print(f"[HTTPServer] Generate exp={eid}: {len(prompt_ids)} prompts -> {len(comp_ids)} completions, {gen_ms:.0f}ms")

        return _msgpack_response({
            "completion_texts": comp_texts,
            "completion_ids": comp_ids,
            "prompt_ids": prompt_ids_out,
        })

    @app.post("/set_scales")
    async def set_scales(request: Request):
        body = await request.body()
        msg = msgpack.unpackb(body, raw=False)
        state["mgr"].set_scales(
            msg["experiment_id"],
            msg["retain_scale"],
            msg["forget_scale"],
        )
        return _msgpack_response({"ok": True})

    @app.post("/shutdown")
    def shutdown():
        print("[HTTPServer] Shutdown requested")
        def _shutdown():
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=_shutdown, daemon=True).start()
        return _msgpack_response({"ok": True})

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="HTTP vLLM generation server with MLP adapters")
    parser.add_argument("--model", default="SimpleStories/SimpleStories-1.25M",
                        help="HuggingFace model name")
    parser.add_argument("--max_experiments", type=int, default=10)
    parser.add_argument("--mlp_config", default="m16", choices=list(MLP_PRESETS.keys()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.05)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    preset = MLP_PRESETS[args.mlp_config]

    app = create_app(
        model_name=args.model,
        max_experiments=args.max_experiments,
        retain_neurons=preset["retain_neurons"],
        forget_neurons=preset["forget_neurons"],
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )

    print(f"[HTTPServer] Starting on http://{args.host}:{args.port}")
    # Single worker to ensure sequential request handling (sync vLLM engine
    # isn't thread-safe). Training clients overlap their HF forward/backward
    # with other clients' server requests.
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")


if __name__ == "__main__":
    main()
