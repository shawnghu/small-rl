"""vLLM generation server for client-server GRPO training.

Hosts a shared vLLM engine with N adapter slots. Clients connect via ZMQ IPC
and send weight updates + generation requests.

Usage (standalone, for debugging):
    CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        .venv-vllm/bin/python vllm_server.py --max_experiments 10 --mlp_config m16
"""

import argparse
import os
import time

import msgpack
import numpy as np
import torch
import zmq
from vllm import SamplingParams

from vllm_grpo import MLP_PRESETS, flatten_vllm_outputs
from vllm_mlp_adapter import create_engine

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Weight tensor names in order (must match client)
WEIGHT_KEYS = [
    "gate_retain", "up_retain", "down_retain",
    "gate_forget", "up_forget", "down_forget",
]


class VLLMServer:
    """ZMQ-based vLLM generation server with per-experiment adapter routing."""

    def __init__(self, socket_addr, max_experiments, retain_neurons, forget_neurons,
                 model_name=MODEL_NAME, gpu_memory_utilization=0.05,
                 layer_start=0.0, layer_end=1.0, layer_stride=1):
        self.socket_addr = socket_addr
        self.max_experiments = max_experiments

        # Create vLLM engine + adapter manager
        print(f"[Server] Creating vLLM engine (max_experiments={max_experiments}, "
              f"retain={retain_neurons}, forget={forget_neurons}, "
              f"layers={layer_start:.2f}-{layer_end:.2f})...")
        t0 = time.time()
        self.llm, self.mgr = create_engine(
            model_name=model_name,
            max_experiments=max_experiments,
            retain_neurons=retain_neurons,
            forget_neurons=forget_neurons,
            gpu_memory_utilization=gpu_memory_utilization,
            layer_start=layer_start,
            layer_end=layer_end,
            layer_stride=layer_stride,
        )
        print(f"[Server] Engine ready in {time.time() - t0:.1f}s")

        # ZMQ setup
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        # Clean up stale socket file if it exists
        if socket_addr.startswith("ipc://"):
            sock_path = socket_addr[len("ipc://"):]
            if os.path.exists(sock_path):
                os.unlink(sock_path)
        self.socket.bind(socket_addr)
        print(f"[Server] Listening on {socket_addr}")

        self.next_experiment_id = 1

    def handle_register(self, msg):
        eid = self.next_experiment_id
        assert eid <= self.max_experiments, \
            f"Cannot register: {eid} > max_experiments={self.max_experiments}"
        self.next_experiment_id += 1
        print(f"[Server] Registered experiment {eid}")
        return {"experiment_id": eid}

    def handle_update_weights(self, msg):
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

        self.mgr.set_weights(eid, layer_weights)
        return {"ok": True}

    def handle_generate(self, msg):
        eid = msg["experiment_id"]
        prompt_ids = msg["prompt_ids"]
        top_k = msg.get("top_k", 50)
        top_p = msg.get("top_p", 1.0)
        return_logprobs = msg.get("return_logprobs", False)
        sp = SamplingParams(
            n=msg["n"],
            temperature=msg["temperature"],
            max_tokens=msg["max_tokens"],
            top_k=top_k if top_k > 0 else -1,
            top_p=top_p,
            logprobs=0 if return_logprobs else None,
        )

        outputs = self.mgr.generate(prompt_ids, [eid] * len(prompt_ids), sp)
        comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(outputs)

        reply = {
            "completion_texts": comp_texts,
            "completion_ids": comp_ids,
            "prompt_ids": prompt_ids_out,
        }

        if return_logprobs:
            # Extract per-token logprobs for each completion
            all_logprobs = []
            for req in outputs:
                for comp in req.outputs:
                    token_logprobs = []
                    for i, lp_dict in enumerate(comp.logprobs):
                        tid = comp.token_ids[i]
                        entry = lp_dict.get(tid)
                        token_logprobs.append(entry.logprob if entry is not None else 0.0)
                    all_logprobs.append(token_logprobs)
            reply["logprobs"] = all_logprobs

        return reply

    def handle_set_scales(self, msg):
        self.mgr.set_scales(
            msg["experiment_id"],
            msg["retain_scale"],
            msg["forget_scale"],
        )
        return {"ok": True}

    def handle_release(self, msg):
        # No-op for now: with per-run servers (max_experiments=1), there's
        # nothing to clean up. Multi-experiment servers could zero the slot.
        return {"ok": True}

    def run(self, ready_event=None):
        """Main server loop. Set ready_event when ready to accept clients."""
        if ready_event is not None:
            ready_event.set()

        print("[Server] Ready for requests")
        while True:
            raw = self.socket.recv()
            msg = msgpack.unpackb(raw, raw=False)
            op = msg["op"]

            try:
                if op == "register":
                    reply = self.handle_register(msg)
                elif op == "update_weights":
                    reply = self.handle_update_weights(msg)
                elif op == "generate":
                    reply = self.handle_generate(msg)
                elif op == "set_scales":
                    reply = self.handle_set_scales(msg)
                elif op == "release":
                    reply = self.handle_release(msg)
                elif op == "shutdown":
                    self.socket.send(msgpack.packb({"ok": True}, use_bin_type=True))
                    print("[Server] Shutting down")
                    break
                else:
                    reply = {"error": f"unknown op: {op}"}
            except Exception as e:
                reply = {"error": str(e)}
                import traceback
                traceback.print_exc()

            self.socket.send(msgpack.packb(reply, use_bin_type=True))

        self.ctx.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM generation server")
    parser.add_argument("--socket", default="ipc:///tmp/vllm_grpo.sock")
    parser.add_argument("--max_experiments", type=int, default=10)
    parser.add_argument("--mlp_config", default="m16", choices=list(MLP_PRESETS.keys()))
    return parser.parse_args()


def main():
    args = parse_args()
    preset = MLP_PRESETS[args.mlp_config]
    server = VLLMServer(
        socket_addr=args.socket,
        max_experiments=args.max_experiments,
        retain_neurons=preset["retain_neurons"],
        forget_neurons=preset["forget_neurons"],
    )
    server.run()


if __name__ == "__main__":
    main()
