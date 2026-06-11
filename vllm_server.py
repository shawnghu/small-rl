"""vLLM generation server for client-server GRPO training.

Hosts a shared vLLM engine with N adapter slots. Clients connect via ZMQ IPC
and send weight updates + generation requests.

Usage (standalone, for debugging):
    CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        .venv/bin/python vllm_server.py --model HuggingFaceTB/SmolLM2-135M-Instruct --max_experiments 10 --mlp_config m16
"""

import argparse
import os
import time

import msgpack
import zmq
from vllm import SamplingParams

from vllm_utils import MLP_PRESETS, deserialize_layer_weights, flatten_vllm_outputs
from vllm_mlp_adapter import create_engine


class VLLMServer:
    """ZMQ-based vLLM generation server with per-experiment adapter routing."""

    def __init__(self, socket_addr, max_experiments, retain_neurons, forget_neurons,
                 model_name, gpu_memory_utilization=0.05, dtype="bfloat16",
                 layer_start=0.0, layer_end=1.0, layer_stride=1, enforce_eager=True,
                 max_num_seqs=None):
        self.socket_addr = socket_addr
        self.max_experiments = max_experiments

        # Create vLLM engine + adapter manager
        print(f"[Server] Creating vLLM engine (max_experiments={max_experiments}, "
              f"retain={retain_neurons}, forget={forget_neurons}, dtype={dtype}, "
              f"layers={layer_start:.2f}-{layer_end:.2f})...")
        t0 = time.time()
        self.llm, self.mgr = create_engine(
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            model_name=model_name,
            max_experiments=max_experiments,
            retain_neurons=retain_neurons,
            forget_neurons=forget_neurons,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            layer_start=layer_start,
            layer_end=layer_end,
            layer_stride=layer_stride,
        )
        print(f"[Server] Engine ready in {time.time() - t0:.1f}s")

        # Optional per-phase engine instrumentation (env VLLM_ENGINE_TIMERS=1):
        # schedule / exec_wait / update / proc_out cumulative wall time per
        # generate, reported in the generate reply's timings dict.
        self._engine_timers = None
        if os.environ.get("VLLM_ENGINE_TIMERS") == "1":
            import vllm_engine_timers
            vllm_engine_timers.install(self.llm)
            self._engine_timers = vllm_engine_timers
            print("[Server] Engine phase timers installed")

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
        if "flat" in msg:
            # Flat bf16 fast path (sync client): one buffer, in-place slot
            # update + prefix-cache reset inside set_weights_flat.
            import numpy as np
            import torch
            flat = torch.from_numpy(
                np.frombuffer(msg["flat"], dtype=np.uint8).copy()
            ).view(torch.bfloat16)
            self.mgr.set_weights_flat(eid, flat, msg["flat_shapes"])
        else:
            # Legacy per-tensor dict protocol (async client still uses it).
            layer_weights = deserialize_layer_weights(msg)
            self.mgr.set_weights(eid, layer_weights)
        return {"ok": True}

    def handle_generate(self, msg):
        prompt_ids = msg["prompt_ids"]
        # Accept either scalar experiment_id (applied to all prompts) or per-prompt
        # experiment_ids list (for concurrent multi-adapter eval).
        if "experiment_ids" in msg:
            experiment_ids = list(msg["experiment_ids"])
            assert len(experiment_ids) == len(prompt_ids), \
                f"experiment_ids length {len(experiment_ids)} != prompt_ids length {len(prompt_ids)}"
        else:
            experiment_ids = [msg["experiment_id"]] * len(prompt_ids)
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
            # Per-seq-per-step incremental detokenization is pure CPU cost when
            # the caller only needs token ids (training rollouts).
            detokenize=msg.get("detokenize", True),
        )

        import time as _time
        if self._engine_timers is not None:
            self._engine_timers.reset()
        _tg = _time.perf_counter()
        outputs = self.mgr.generate(prompt_ids, experiment_ids, sp)
        _gen_s = _time.perf_counter() - _tg
        _tf = _time.perf_counter()
        comp_texts, comp_ids, prompt_ids_out, _ = flatten_vllm_outputs(outputs)
        _flatten_s = _time.perf_counter() - _tf

        reply = {
            "completion_texts": comp_texts,
            "completion_ids": comp_ids,
            "prompt_ids": prompt_ids_out,
            # Direct gen-phase timing decomposition (serial-bottleneck hunt):
            # add_request / engine.step / output-collect from the manager loop,
            # plus flatten. Client stashes as _last_gen_timings.
            "timings": {
                **getattr(self.mgr, "_last_gen_timings", {}),
                "mgr_generate_s": round(_gen_s, 4),
                "flatten_s": round(_flatten_s, 4),
                **(self._engine_timers.snapshot() if self._engine_timers is not None else {}),
            },
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

    def handle_sleep(self, msg):
        """Put vLLM engine to sleep: offload weights to CPU, discard KV cache."""
        level = msg.get("level", 1)
        self.llm.sleep(level=level)
        print(f"[Server] Engine sleeping (level={level})")
        return {"ok": True}

    def handle_wake_up(self, msg):
        """Wake vLLM engine: reload weights and reallocate KV cache."""
        tags = msg.get("tags", None)
        self.llm.wake_up(tags=tags)
        print(f"[Server] Engine awake (tags={tags})")
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
                elif op == "sleep":
                    reply = self.handle_sleep(msg)
                elif op == "wake_up":
                    reply = self.handle_wake_up(msg)
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
    parser.add_argument("--model", required=True, help="HuggingFace model name")
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
        model_name=args.model,
    )
    server.run()


if __name__ == "__main__":
    main()
