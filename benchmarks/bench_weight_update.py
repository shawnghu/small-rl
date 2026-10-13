"""Benchmark: decompose the cost of an async (AsyncLLM) weight push.

Context: the async MLP server path (AsyncVLLMAdapterManager + inject_mlp_adapters)
is currently bitrotted (references a deleted _inject_routing_hook), so we can't
time it end-to-end. But the question that matters for a one-step-off / multiplexed
async design is: what does the *mechanism* cost — `engine.collective_rpc(...)`
round-trips + shipping the adapter tensors across the EngineCoreProc boundary +
reset_prefix_cache — independent of the (broken) adapter injection.

So we time it on a BARE AsyncLLM engine (no adapter injection):

  A. collective_rpc(noop)            — fixed RPC round-trip floor (closure pickle + IPC)
  B. collective_rpc(copy ~Nmb)       — A + serialize/ship ~180 tensors (~25MB f32) + H2D
  C. reset_prefix_cache()            — the second round-trip async set_weights also does

The async set_weights = B + C. The sync fast path (set_weights_flat) does the
equivalent as one in-process H2D with NO pickling/IPC/round-trip — so the gap
B+C minus (a few ms of H2D) is the async process-boundary tax.

Run (single line):
    CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv/bin/python -u benchmarks/bench_weight_update.py
"""

import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
GPU_MEM = 0.1
RETAIN = 32
FORGET = 32
N_WARMUP = 3
N_TRIALS = 12


def _make_payload(num_layers, hidden, retain, forget):
    """~180 CPU f32 tensors matching the real adapter update shapes/bytes."""
    import torch
    tensors = []
    for _ in range(num_layers):
        if retain > 0:
            tensors += [torch.randn(retain, hidden), torch.randn(retain, hidden),
                        torch.randn(hidden, retain)]
        if forget > 0:
            tensors += [torch.randn(forget, hidden), torch.randn(forget, hidden),
                        torch.randn(hidden, forget)]
    nbytes = sum(t.numel() * 4 for t in tensors)
    return tensors, nbytes


def _run(result_queue):
    import asyncio
    import torch
    from transformers import AutoConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    cfg = AutoConfig.from_pretrained(MODEL)
    hidden, num_layers = cfg.hidden_size, cfg.num_hidden_layers
    payload, nbytes = _make_payload(num_layers, hidden, RETAIN, FORGET)

    def med(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    async def run():
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=MODEL, enforce_eager=True, dtype="bfloat16",
            gpu_memory_utilization=GPU_MEM,
        ))

        def _noop(model):
            return None

        def _copy(model):
            dev = next(model.parameters()).device
            s = 0
            for t in payload:
                s += t.to(dev, non_blocking=True).sum()
            torch.cuda.synchronize()
            return None

        async def timeit(factory):
            for _ in range(N_WARMUP):
                await factory()
            ts = []
            for _ in range(N_TRIALS):
                t0 = time.perf_counter()
                await factory()
                ts.append((time.perf_counter() - t0) * 1000)
            return ts

        results = {}
        print(f"[bench] payload: {len(payload)} tensors, {nbytes/1e6:.1f} MB f32", flush=True)
        print("[bench] A: collective_rpc(noop)...", flush=True)
        results["A_rpc_noop"] = await timeit(
            lambda: engine.collective_rpc("apply_model", args=(_noop,)))
        print("[bench] B: collective_rpc(copy payload)...", flush=True)
        results["B_rpc_copy"] = await timeit(
            lambda: engine.collective_rpc("apply_model", args=(_copy,)))
        print("[bench] C: reset_prefix_cache()...", flush=True)
        results["C_reset_prefix"] = await timeit(
            lambda: engine.reset_prefix_cache())

        print("[bench] shutting down...", flush=True)
        engine.shutdown()
        return {k: (round(med(v), 1), [round(x, 1) for x in v]) for k, v in results.items()}

    try:
        result_queue.put(asyncio.run(run()))
    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put(e)


def main():
    print(f"Model: {MODEL} | retain={RETAIN} forget={FORGET} | trials={N_TRIALS} (+{N_WARMUP} warmup)\n")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run, args=(q,))
    p.start()
    p.join(timeout=600)
    if p.is_alive():
        p.kill()
        print("TIMEOUT")
        return
    res = q.get()
    if isinstance(res, Exception):
        print("ERROR:", res)
        return
    order = ["A_rpc_noop", "B_rpc_copy", "C_reset_prefix"]
    labels = {
        "A_rpc_noop": "collective_rpc(noop)        [RPC round-trip floor]",
        "B_rpc_copy": "collective_rpc(copy ~25MB)  [+ pickle/ship 180 tensors + H2D]",
        "C_reset_prefix": "reset_prefix_cache()        [2nd round-trip]",
    }
    print(f"\n{'phase':<52}{'median (ms)':>12}")
    print("-" * 64)
    for k in order:
        print(f"{labels[k]:<52}{res[k][0]:>12}")
    bc = res["B_rpc_copy"][0] + res["C_reset_prefix"][0]
    print("-" * 64)
    print(f"{'async set_weights ~= B + C':<52}{round(bc,1):>12}")
    print("\nraw trials (ms):")
    for k in order:
        print(f"  {k:<16}{res[k][1]}")


if __name__ == "__main__":
    main()
