"""vLLM generation server for client-server GRPO training.

Hosts a shared vLLM engine with N adapter slots. Clients connect via ZMQ IPC
and send weight updates + generation requests.

Two server classes:
  - VLLMServer: ZMQ REP socket, serial request handling (one client at a time)
  - BatchingVLLMServer: ZMQ ROUTER socket, accumulates generation requests from
    multiple clients and fires them in a single batched LLM.generate() call

Usage (standalone, for debugging):
    CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        .venv-vllm/bin/python vllm_server.py --max_experiments 10 --mlp_config m16

Usually launched by vllm_client_server_train.py, not directly.
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

MODEL_NAME = "SimpleStories/SimpleStories-1.25M"

# Weight tensor names in order (must match client)
WEIGHT_KEYS = [
    "gate_retain", "up_retain", "down_retain",
    "gate_forget", "up_forget", "down_forget",
]


class VLLMServer:
    """ZMQ-based vLLM generation server with per-experiment adapter routing."""

    def __init__(self, socket_addr, max_experiments, retain_neurons, forget_neurons,
                 model_name=MODEL_NAME, gpu_memory_utilization=0.05):
        self.socket_addr = socket_addr
        self.max_experiments = max_experiments

        # Create vLLM engine + adapter manager
        print(f"[Server] Creating vLLM engine (max_experiments={max_experiments}, "
              f"retain={retain_neurons}, forget={forget_neurons})...")
        t0 = time.time()
        self.llm, self.mgr = create_engine(
            model_name=model_name,
            max_experiments=max_experiments,
            retain_neurons=retain_neurons,
            forget_neurons=forget_neurons,
            gpu_memory_utilization=gpu_memory_utilization,
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
        sp = SamplingParams(
            n=msg["n"],
            temperature=msg["temperature"],
            max_tokens=msg["max_tokens"],
        )

        outputs = self.mgr.generate(prompt_ids, [eid] * len(prompt_ids), sp)
        comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(outputs)

        return {
            "completion_texts": comp_texts,
            "completion_ids": comp_ids,
            "prompt_ids": prompt_ids_out,
        }

    def handle_set_scales(self, msg):
        self.mgr.set_scales(
            msg["experiment_id"],
            msg["retain_scale"],
            msg["forget_scale"],
        )
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


class BatchingVLLMServer:
    """ZMQ ROUTER-based batching vLLM server.

    DEPRECATED: Initial experiments showed no throughput improvement over the
    serial VLLMServer. The batching happens at the request level (not inside
    vLLM), so vLLM's own continuous batching already handles this. May revisit
    if cross-experiment batching becomes important.

    Accumulates generation requests from multiple concurrent clients and fires
    them in a single batched LLM.generate() call. Non-generate ops (register,
    weight update, set_scales, shutdown) are handled immediately.

    Uses the sync VLLMAdapterManager/LLM engine — the batching is done at the
    request level, not inside vLLM. This avoids the async engine's n>1 bug and
    per-request overhead while getting full cross-experiment batching.
    """

    # Max time (ms) to wait for more generate requests before firing a batch
    BATCH_TIMEOUT_MS = 50

    def __init__(self, socket_addr, max_experiments, retain_neurons, forget_neurons,
                 model_name=MODEL_NAME, gpu_memory_utilization=0.05):
        self.socket_addr = socket_addr
        self.max_experiments = max_experiments

        # Create vLLM engine + adapter manager (sync)
        print(f"[BatchServer] Creating vLLM engine (max_experiments={max_experiments}, "
              f"retain={retain_neurons}, forget={forget_neurons})...")
        t0 = time.time()
        self.llm, self.mgr = create_engine(
            model_name=model_name,
            max_experiments=max_experiments,
            retain_neurons=retain_neurons,
            forget_neurons=forget_neurons,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        print(f"[BatchServer] Engine ready in {time.time() - t0:.1f}s")

        # ZMQ ROUTER socket — handles concurrent clients via identity frames
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        if socket_addr.startswith("ipc://"):
            sock_path = socket_addr[len("ipc://"):]
            if os.path.exists(sock_path):
                os.unlink(sock_path)
        self.socket.bind(socket_addr)
        print(f"[BatchServer] Listening on {socket_addr}")

        self.next_experiment_id = 1
        self._n_registered = 0
        self._shutdown = False

    def _send_reply(self, identity, reply):
        """Send a msgpack reply to a specific client."""
        self.socket.send_multipart([
            identity, b"",
            msgpack.packb(reply, use_bin_type=True),
        ])

    def _handle_non_generate(self, identity, msg):
        """Handle a non-generate op immediately and send reply."""
        op = msg["op"]
        try:
            if op == "register":
                eid = self.next_experiment_id
                assert eid <= self.max_experiments, \
                    f"Cannot register: {eid} > max_experiments={self.max_experiments}"
                self.next_experiment_id += 1
                self._n_registered += 1
                print(f"[BatchServer] Registered experiment {eid} "
                      f"({self._n_registered} total)")
                reply = {"experiment_id": eid}
            elif op == "update_weights":
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
                reply = {"ok": True}
            elif op == "set_scales":
                self.mgr.set_scales(
                    msg["experiment_id"],
                    msg["retain_scale"],
                    msg["forget_scale"],
                )
                reply = {"ok": True}
            elif op == "shutdown":
                self._shutdown = True
                reply = {"ok": True}
            else:
                reply = {"error": f"unknown op: {op}"}
        except Exception as e:
            reply = {"error": str(e)}
            import traceback
            traceback.print_exc()

        self._send_reply(identity, reply)

    def _fire_batch(self, pending):
        """Merge pending generate requests into one LLM.generate() call."""
        from vllm import TokensPrompt
        from vllm.lora.request import LoRARequest

        # Build merged prompt list + per-prompt metadata
        all_prompts = []
        all_sps = []
        all_lora_reqs = []
        # Track (identity, prompt_start, prompt_end) for result scattering
        client_ranges = []

        offset = 0
        for identity, msg in pending:
            eid = msg["experiment_id"]
            n_prompts = len(msg["prompt_ids"])
            sp = SamplingParams(
                n=msg["n"],
                temperature=msg["temperature"],
                max_tokens=msg["max_tokens"],
            )
            lora_req = LoRARequest(
                lora_name=f"experiment_{eid}",
                lora_int_id=eid,
                lora_path=self.mgr._lora_dir,
            )
            for pid in msg["prompt_ids"]:
                all_prompts.append(TokensPrompt(prompt_token_ids=list(pid)))
                all_sps.append(sp)
                all_lora_reqs.append(lora_req)

            client_ranges.append((identity, offset, offset + n_prompts))
            offset += n_prompts

        total_prompts = len(all_prompts)
        n_clients = len(pending)
        t0 = time.time()

        try:
            outputs = self.llm.generate(
                all_prompts, all_sps, lora_request=all_lora_reqs,
            )
            gen_ms = (time.time() - t0) * 1000
            print(f"[BatchServer] Batched generate: {n_clients} clients, "
                  f"{total_prompts} prompts, {gen_ms:.0f}ms")

            # Scatter results back to each client
            for identity, start, end in client_ranges:
                client_outputs = outputs[start:end]
                comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(
                    client_outputs)
                self._send_reply(identity, {
                    "completion_texts": comp_texts,
                    "completion_ids": comp_ids,
                    "prompt_ids": prompt_ids_out,
                })

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_reply = {"error": str(e)}
            for identity, _, _ in client_ranges:
                self._send_reply(identity, error_reply)

    def _drain_messages(self, pending):
        """Non-blocking drain of all available messages from the socket.

        Generate ops are appended to `pending`. Non-generate ops are handled
        immediately (reply sent inline).
        """
        while True:
            try:
                frames = self.socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            identity = frames[0]
            payload = frames[-1]
            msg = msgpack.unpackb(payload, raw=False)

            if msg["op"] == "generate":
                pending.append((identity, msg))
            else:
                self._handle_non_generate(identity, msg)

    def _handle_generate(self, identity, msg):
        """Handle a single generate request immediately."""
        eid = msg["experiment_id"]
        sp = SamplingParams(
            n=msg["n"],
            temperature=msg["temperature"],
            max_tokens=msg["max_tokens"],
        )
        t0 = time.time()
        try:
            outputs = self.mgr.generate(
                msg["prompt_ids"], [eid] * len(msg["prompt_ids"]), sp)
            gen_ms = (time.time() - t0) * 1000
            comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(outputs)
            print(f"[BatchServer] Generate exp={eid}: "
                  f"{len(msg['prompt_ids'])} prompts, {gen_ms:.0f}ms")
            self._send_reply(identity, {
                "completion_texts": comp_texts,
                "completion_ids": comp_ids,
                "prompt_ids": prompt_ids_out,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_reply(identity, {"error": str(e)})

    def run(self, ready_event=None):
        """Main server loop — handles each request immediately (FIFO).

        Clients overlap their training (HF forward/backward via MPS) with
        other clients' generation requests on the server.
        """
        if ready_event is not None:
            ready_event.set()

        print("[BatchServer] Ready for requests")

        while not self._shutdown:
            try:
                frames = self.socket.recv_multipart()
            except zmq.ZMQError:
                break
            identity = frames[0]
            payload = frames[-1]
            msg = msgpack.unpackb(payload, raw=False)

            if msg["op"] == "generate":
                self._handle_generate(identity, msg)
            else:
                self._handle_non_generate(identity, msg)

        print("[BatchServer] Shutting down")
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
