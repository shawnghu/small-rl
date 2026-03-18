"""Async vLLM generation server with dynamic batching across experiments.

Uses AsyncLLM so that generation requests from multiple experiments are
dynamically batched by the engine core's continuous batching scheduler.
This is the key difference from vllm_server.py (which uses synchronous LLM
and serializes generation requests).

Why not `vllm serve`? The OpenAI-compatible server doesn't expose apply_model()
or collective_rpc() with arbitrary Python callables. Our MLP adapter injection
requires passing a function into the worker process that replaces model layers
with VLLMDualMLPAdapter modules, and weight updates push raw tensors into those
custom modules. This can't be serialized over HTTP/JSON. The /collective_rpc
endpoint (dev mode only) only accepts string method names, not callables.

Architecture:
    - AsyncLLM engine runs in a background process (EngineCoreProc)
    - ZMQ ROUTER socket handles concurrent client requests
    - Generation requests from different experiments enter the engine core's
      scheduler and get dynamically batched in the same forward pass
    - Weight updates go through collective_rpc (processed between engine steps)

Usage (standalone):
    CUDA_VISIBLE_DEVICES=1 .venv-vllm-017/bin/python vllm_async_server.py --max_experiments 10 --mlp_config m16

Usually launched by a training orchestrator, not directly.
"""

import argparse
import asyncio
import os
import time

import msgpack
import numpy as np
import torch
import zmq
import zmq.asyncio
from vllm import SamplingParams

from vllm_grpo import MLP_PRESETS, flatten_vllm_outputs
from vllm_mlp_adapter import create_async_engine

MODEL_NAME = "SimpleStories/SimpleStories-1.25M"

WEIGHT_KEYS = [
    "gate_retain", "up_retain", "down_retain",
    "gate_forget", "up_forget", "down_forget",
]


class AsyncVLLMServer:
    """ZMQ ROUTER-based async vLLM server with dynamic batching."""

    def __init__(self, socket_addr, max_experiments, retain_neurons, forget_neurons,
                 model_name=MODEL_NAME, gpu_memory_utilization=0.05):
        self.socket_addr = socket_addr
        self.max_experiments = max_experiments
        self.retain_neurons = retain_neurons
        self.forget_neurons = forget_neurons
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization

        self.engine = None
        self.mgr = None
        self._shutdown = False
        # Slot pool: experiment IDs are recycled rather than allocated forever.
        # max_experiments = concurrent runs on this GPU (not total lifetime runs).
        # _slot_queue is pre-populated in start(); register awaits a slot,
        # release zeros weights then returns the slot. Both run as tasks so
        # the server loop never blocks and there's no deadlock when all slots
        # are occupied and releases are in-flight.
        self._slot_queue: asyncio.Queue | None = None  # created in start()
        # Serializes collective_rpc calls: vLLM's engine uses a single ZMQ
        # socket for collective_rpc without per-request IDs, so concurrent
        # awaiting coroutines would receive each other's responses. The lock
        # lets the server loop stay unblocked (generate tasks can proceed)
        # while ensuring only one collective_rpc is in-flight at a time.
        self._rpc_lock: asyncio.Lock | None = None  # created in start()

    async def start(self):
        """Initialize engine and ZMQ socket."""
        print(f"[AsyncServer] Creating async vLLM engine "
              f"(max_experiments={self.max_experiments}, "
              f"retain={self.retain_neurons}, forget={self.forget_neurons})...")
        t0 = time.time()
        self.engine, self.mgr = await create_async_engine(
            model_name=self.model_name,
            max_experiments=self.max_experiments,
            retain_neurons=self.retain_neurons,
            forget_neurons=self.forget_neurons,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        print(f"[AsyncServer] Engine ready in {time.time() - t0:.1f}s")

        # ZMQ async ROUTER socket — handles concurrent clients
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.ROUTER)
        if self.socket_addr.startswith("ipc://"):
            sock_path = self.socket_addr[len("ipc://"):]
            if os.path.exists(sock_path):
                os.unlink(sock_path)
        self.socket.bind(self.socket_addr)
        print(f"[AsyncServer] Listening on {self.socket_addr}")
        self._rpc_lock = asyncio.Lock()
        self._slot_queue = asyncio.Queue()
        for slot in range(1, self.max_experiments + 1):
            self._slot_queue.put_nowait(slot)

    async def handle_register(self, msg):
        """Acquire a free slot from the pool. Blocks (as a task) until one is available."""
        slot = await self._slot_queue.get()
        n_free = self._slot_queue.qsize()
        print(f"[AsyncServer] Registered experiment {slot} ({n_free} slots remaining)")
        return {"experiment_id": slot}

    async def handle_release(self, msg):
        """Zero the slot's weights then return it to the pool."""
        eid = msg["experiment_id"]
        async with self._rpc_lock:
            await self.mgr.reset_weights(eid)
        self._slot_queue.put_nowait(eid)
        n_free = self._slot_queue.qsize()
        print(f"[AsyncServer] Released experiment {eid} ({n_free} slots free)")
        return {"ok": True}

    async def handle_update_weights(self, msg):
        eid = msg["experiment_id"]
        dtype_str = msg["dtype"]
        np_dtype = np.float32 if dtype_str == "float32" else np.float16

        layer_weights = []
        for layer_data in msg["layers"]:
            w = {}
            for key in WEIGHT_KEYS:
                raw = layer_data.get(key)
                if raw is not None:
                    shape = tuple(msg["shapes"][key])
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                    w[key] = torch.from_numpy(arr.copy())
            layer_weights.append(w)

        async with self._rpc_lock:
            await self.mgr.set_weights(eid, layer_weights)
        return {"ok": True}

    async def handle_generate(self, msg):
        """Submit generation request — dynamically batched by AsyncLLM."""
        eid = msg["experiment_id"]
        prompt_ids = msg["prompt_ids"]
        sp = SamplingParams(
            n=msg["n"],
            temperature=msg["temperature"],
            max_tokens=msg["max_tokens"],
        )

        outputs = await self.mgr.generate(prompt_ids, [eid] * len(prompt_ids), sp)
        comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(outputs)

        return {
            "completion_texts": comp_texts,
            "completion_ids": comp_ids,
            "prompt_ids": prompt_ids_out,
        }

    async def handle_set_scales(self, msg):
        await self.mgr.set_scales(
            msg["experiment_id"],
            msg["retain_scale"],
            msg["forget_scale"],
        )
        return {"ok": True}

    async def _handle_request(self, identity, msg):
        """Handle a single client request. Runs as a concurrent task."""
        op = msg["op"]
        try:
            if op == "register":
                reply = await self.handle_register(msg)
            elif op == "release":
                reply = await self.handle_release(msg)
            elif op == "update_weights":
                reply = await self.handle_update_weights(msg)
            elif op == "generate":
                reply = await self.handle_generate(msg)
            elif op == "set_scales":
                reply = await self.handle_set_scales(msg)
            elif op == "shutdown":
                self._shutdown = True
                reply = {"ok": True}
            else:
                reply = {"error": f"unknown op: {op}"}
        except Exception as e:
            reply = {"error": str(e)}
            import traceback
            traceback.print_exc()

        # Send reply back to the specific client (ROUTER routing)
        await self.socket.send_multipart([
            identity, b"",
            msgpack.packb(reply, use_bin_type=True),
        ])

    async def run(self, ready_event=None):
        """Main async server loop."""
        await self.start()
        if ready_event is not None:
            ready_event.set()

        # Track in-flight tasks so we can await them before shutdown.
        pending_tasks: set[asyncio.Task] = set()

        print("[AsyncServer] Ready for requests")
        while not self._shutdown:
            try:
                frames = await asyncio.wait_for(
                    self.socket.recv_multipart(), timeout=0.5,
                )
            except asyncio.TimeoutError:
                continue

            # ROUTER frames: [identity, empty, payload]
            identity = frames[0]
            payload = frames[-1]
            msg = msgpack.unpackb(payload, raw=False)

            op = msg["op"]
            if op in ("generate", "update_weights", "register", "release"):
                # generate: enables dynamic batching — multiple tasks in-flight
                #   simultaneously; AsyncLLM batches their requests.
                # update_weights: ~400ms collective_rpc round-trip; running as a
                #   task lets the loop keep accepting messages (e.g. generate
                #   requests from experiments that already finished updating).
                #   Per-experiment slot isolation makes concurrent updates safe;
                #   collective_rpc calls are serialized by _rpc_lock.
                # register: blocks (as a task) if all slots occupied, so the
                #   server loop stays unblocked while waiting for a free slot.
                # release: zeros weights via collective_rpc then returns slot;
                #   must run as a task so register tasks can unblock once done.
                task = asyncio.create_task(self._handle_request(identity, msg))
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)
            else:
                # set_scales, shutdown: fast and order-sensitive,
                # run inline to preserve ordering guarantees.
                await self._handle_request(identity, msg)

        # Wait for any in-flight weight-update or generate tasks to finish
        # before tearing down the engine, so they don't crash on a dead engine.
        if pending_tasks:
            print(f"[AsyncServer] Waiting for {len(pending_tasks)} in-flight tasks...")
            await asyncio.gather(*pending_tasks, return_exceptions=True)

        print("[AsyncServer] Shutting down")
        self.engine.shutdown()
        self.ctx.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="Async vLLM generation server")
    parser.add_argument("--socket", default="ipc:///tmp/vllm_grpo_async.sock")
    parser.add_argument("--max_experiments", type=int, default=10)
    parser.add_argument("--mlp_config", default="m16", choices=list(MLP_PRESETS.keys()))
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    preset = MLP_PRESETS[args.mlp_config]
    server = AsyncVLLMServer(
        socket_addr=args.socket,
        max_experiments=args.max_experiments,
        retain_neurons=preset["retain_neurons"],
        forget_neurons=preset["forget_neurons"],
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
