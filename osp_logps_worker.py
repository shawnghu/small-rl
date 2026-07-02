"""OSP logps worker: a dedicated process for the rollout-phase old_logps forward.

Under --osp_worker, the rollout thread in the training process stops running the
HF old_logps forward itself (60s+ of GIL-holding kernel-launch loop per cycle
that contends with the concurrent optimizer update) and instead ships the batch
to this worker over ZMQ. The thread's remaining in-process work is decode +
rewards (~1-2s of GIL per cycle); everything else is GIL-free ZMQ blocking.

Process model:
  - Spawned like the per-run vLLM server (spawn context, module-level target,
    parent flips CUDA_VISIBLE_DEVICES around Process.start() so this process
    sees exactly ONE GPU as cuda:0). Single-visible-device by construction:
    no triton cross-device stream hazard, no HF n_gpu batch scaling, no
    device-threading anywhere in this process.
  - Owns a full frozen replica of the policy (base + dual MLP adapters),
    loaded from disk at startup. Per cycle the trainer sends the trainable
    (adapter) params as CPU tensors at the OSP barrier — the same snapshot
    semantics as sync_adapter_snapshot, so old_logps are computed against
    exactly the theta that vLLM samples with.
  - Computes logps through the SAME code path as the in-process path
    (trl_overrides._ref_logps_dynamic -> _ref_logps_liger_fused) via a duck-typed
    trainer shim, so the update/old kernel-consistency contract is preserved by
    construction, not by a parallel implementation.

Protocol (pickle over ZMQ REQ/REP; one client, lockstep with the OSP barrier):
  {"op": "sync_adapters", "state": {name: cpu_tensor}}      -> {"ok": True}
  {"op": "logps", "ids": cpu_long, "mask": cpu_long,
   "logits_to_keep": int}                                   -> {"logps": cpu_float, "fwd_s": float}
  {"op": "ping"}                                            -> {"ok": True}
  {"op": "shutdown"}                                        -> {"ok": True} (then exits)
"""

import os
import pickle
import time


def worker_main(model_name, mlp_kwargs, socket_path, ready_file, lock_gpu_id, label,
                log_dir, visible_gpu, temperature, max_tokens_per_bin, bf16=True):
    """Spawn target (module-level for spawn pickling).

    visible_gpu: physical GPU id; the parent also flips CUDA_VISIBLE_DEVICES
    around Process.start() (a spawn child re-imports modules before this
    function runs, which can init CUDA under the parent's multi-GPU view —
    the same gotcha as the vLLM server child). Set here too, belt and braces.
    """
    if visible_gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)

    from vllm_lifecycle import vllm_init_slot, vllm_worker_setup_signals
    vllm_worker_setup_signals()  # setsid -> own process group for killpg cleanup

    if log_dir is not None:
        import sys
        os.makedirs(log_dir, exist_ok=True)
        _fh = open(os.path.join(log_dir, "osp_logps_worker.log"), "w", buffering=1)
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(_fh.fileno(), 1)
        os.dup2(_fh.fileno(), 2)
        sys.stdout = _fh
        sys.stderr = _fh

    import inspect
    import types

    import torch
    import zmq

    # Serialize the CUDA-heavy init against concurrent vLLM init on the same
    # physical GPU (same per-GPU flock the vLLM server child uses).
    with vllm_init_slot(lock_gpu_id, label):
        from transformers import AutoModelForCausalLM
        from gradient_routing import apply_dual_mlp

        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if bf16 else None,
            attn_implementation="flash_attention_2",
        )
        apply_dual_mlp(model, **mlp_kwargs)
        model.cuda().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"[osp_worker] model + adapters loaded in {time.perf_counter() - t0:.1f}s "
              f"on {torch.cuda.get_device_name(0)} (CVD={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    # Duck-typed trainer shim so trl_overrides._ref_logps_dynamic runs its exact
    # in-trainer code path (single source of truth for the old_logps kernel).
    from trl import GRPOTrainer

    class _Acc:
        device = torch.device("cuda:0")
        is_main_process = True

        @staticmethod
        def unwrap_model(m):
            return m

    shim = types.SimpleNamespace(
        accelerator=_Acc(),
        _rollout_model=model,
        model=model,
        temperature=temperature,
        args=types.SimpleNamespace(),
        model_kwarg_keys=set(inspect.signature(model.forward).parameters.keys()),
    )
    shim._get_last_hidden_state = types.MethodType(GRPOTrainer._get_last_hidden_state, shim)

    import trl_overrides

    adapter_params = {n: p for n, p in model.named_parameters()
                      if "_retain" in n or "_forget" in n}

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(socket_path)
    with open(ready_file, "w") as f:
        f.write("ready")
    print(f"[osp_worker] serving on {socket_path}")

    while True:
        msg = pickle.loads(sock.recv())
        try:
            op = msg["op"]
            if op == "ping":
                reply = {"ok": True}
            elif op == "shutdown":
                sock.send(pickle.dumps({"ok": True}))
                break
            elif op == "sync_adapters":
                state = msg["state"]
                # Fail loudly on any mismatch: a silently missing adapter param
                # means old_logps against the wrong theta.
                assert set(state.keys()) == set(adapter_params.keys()), (
                    f"adapter param set mismatch: "
                    f"missing={set(adapter_params) - set(state)} "
                    f"extra={set(state) - set(adapter_params)}")
                with torch.no_grad():
                    for n, t in state.items():
                        adapter_params[n].data.copy_(t, non_blocking=True)
                torch.cuda.synchronize()
                reply = {"ok": True}
            elif op == "logps":
                ids = msg["ids"].cuda(non_blocking=True)
                mask = msg["mask"].cuda(non_blocking=True)
                t0 = time.perf_counter()
                with torch.no_grad():
                    logps = trl_overrides._ref_logps_dynamic(
                        shim, ids, mask, msg["logits_to_keep"],
                        num_images=None,
                        max_tokens_per_bin=max_tokens_per_bin,
                        forward_kwargs={},
                        use_liger_fused=True,
                    )
                torch.cuda.synchronize()
                reply = {"logps": logps.cpu(), "fwd_s": time.perf_counter() - t0}
            else:
                reply = {"error": f"unknown op {op!r}"}
        except Exception as e:  # surface in the trainer, don't hang the REP socket
            import traceback
            traceback.print_exc()
            reply = {"error": f"{type(e).__name__}: {e}"}
        sock.send(pickle.dumps(reply))
    print("[osp_worker] shutdown")


class OSPLogpsClient:
    """Trainer-side client. All calls block (REQ/REP lockstep); the rollout
    thread's blocking recv releases the GIL, which is the point."""

    def __init__(self, socket_path):
        import zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(socket_path)

    def _request(self, msg):
        self._sock.send(pickle.dumps(msg))
        reply = pickle.loads(self._sock.recv())
        if "error" in reply:
            raise RuntimeError(f"osp_logps_worker error: {reply['error']}")
        return reply

    def ping(self):
        return self._request({"op": "ping"})

    def sync_adapters(self, state):
        """state: {param_name: cpu tensor} for every trainable adapter param."""
        return self._request({"op": "sync_adapters", "state": state})

    def logps(self, ids_cpu, mask_cpu, logits_to_keep):
        reply = self._request({"op": "logps", "ids": ids_cpu, "mask": mask_cpu,
                               "logits_to_keep": logits_to_keep})
        self.last_fwd_s = reply["fwd_s"]
        return reply["logps"]

    def shutdown(self):
        try:
            self._request({"op": "shutdown"})
        except Exception:
            pass
