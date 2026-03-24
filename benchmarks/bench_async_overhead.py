"""Benchmark: why is AsyncLLM ~3x slower than sync LLM for the same workload?

Isolates the overhead by testing:
  1. Sync LLM.generate (baseline, uses FINAL_ONLY internally)
  2. Async AsyncLLM with CUMULATIVE output_kind (current default)
  3. Async AsyncLLM with FINAL_ONLY output_kind (the fix)
  4. Sync with VLLM_ENABLE_V1_MULTIPROCESSING=0 (InprocClient, no IPC)
  5. Async with n=1 x 512 prompts vs n=16 x 32 prompts (fan-out overhead)

Each test runs in its own spawned subprocess to avoid CUDA state contamination.

Run:
    CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 .venv-vllm/bin/python benchmarks/bench_async_overhead.py
"""

import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MAX_EXPERIMENTS = 6
RETAIN = 32
FORGET = 32
GPU_MEM = 0.1

N_WARMUP = 1
N_TRIALS = 3
N_PROMPTS = 32
N_COMPLETIONS = 16
MAX_TOKENS = 16
TEMPERATURE = 1.0


def _run_in_subprocess(fn_name, result_queue):
    """Run a named benchmark function in a clean subprocess."""
    import asyncio
    import gc

    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok.encode("Once upon a time", add_special_tokens=False)

    def make_prompts(n):
        return [TokensPrompt(prompt_token_ids=list(prompt_ids)) for _ in range(n)]

    try:
        if fn_name == "sync":
            from vllm_mlp_adapter import create_engine
            llm, mgr = create_engine(
                model_name=MODEL, max_experiments=MAX_EXPERIMENTS,
                retain_neurons=RETAIN, forget_neurons=FORGET,
                gpu_memory_utilization=GPU_MEM,
            )
            sp = SamplingParams(n=N_COMPLETIONS, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            prompts = make_prompts(N_PROMPTS)
            for _ in range(N_WARMUP):
                mgr.generate(prompts, [1] * N_PROMPTS, sp)
            times = []
            for _ in range(N_TRIALS):
                t0 = time.perf_counter()
                results = mgr.generate(prompts, [1] * N_PROMPTS, sp)
                times.append(time.perf_counter() - t0)
                total = sum(len(r.outputs) for r in results)
                assert total == N_PROMPTS * N_COMPLETIONS, f"got {total}"
            del llm, mgr
            gc.collect()
            result_queue.put(times)

        elif fn_name == "sync_inproc":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            from vllm_mlp_adapter import create_engine
            llm, mgr = create_engine(
                model_name=MODEL, max_experiments=MAX_EXPERIMENTS,
                retain_neurons=RETAIN, forget_neurons=FORGET,
                gpu_memory_utilization=GPU_MEM,
            )
            sp = SamplingParams(n=N_COMPLETIONS, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            prompts = make_prompts(N_PROMPTS)
            for _ in range(N_WARMUP):
                mgr.generate(prompts, [1] * N_PROMPTS, sp)
            times = []
            for _ in range(N_TRIALS):
                t0 = time.perf_counter()
                results = mgr.generate(prompts, [1] * N_PROMPTS, sp)
                times.append(time.perf_counter() - t0)
                total = sum(len(r.outputs) for r in results)
                assert total == N_PROMPTS * N_COMPLETIONS, f"got {total}"
            del llm, mgr
            gc.collect()
            result_queue.put(times)

        elif fn_name in ("async_cumulative", "async_final_only"):
            from vllm.sampling_params import RequestOutputKind
            from vllm_mlp_adapter import create_async_engine

            async def run():
                engine, mgr = await create_async_engine(
                    model_name=MODEL, max_experiments=MAX_EXPERIMENTS,
                    retain_neurons=RETAIN, forget_neurons=FORGET,
                    gpu_memory_utilization=GPU_MEM,
                )
                sp = SamplingParams(n=N_COMPLETIONS, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
                if fn_name == "async_final_only":
                    sp.output_kind = RequestOutputKind.FINAL_ONLY
                prompts = make_prompts(N_PROMPTS)
                for _ in range(N_WARMUP):
                    await mgr.generate(prompts, [1] * N_PROMPTS, sp)
                times = []
                for _ in range(N_TRIALS):
                    t0 = time.perf_counter()
                    results = await mgr.generate(prompts, [1] * N_PROMPTS, sp)
                    times.append(time.perf_counter() - t0)
                    total = sum(len(r.outputs) for r in results)
                    assert total == N_PROMPTS * N_COMPLETIONS, f"got {total}"
                engine.shutdown()
                del engine, mgr
                gc.collect()
                return times

            result_queue.put(asyncio.run(run()))

        elif fn_name == "async_n1":
            from vllm.sampling_params import RequestOutputKind
            from vllm_mlp_adapter import create_async_engine

            async def run():
                engine, mgr = await create_async_engine(
                    model_name=MODEL, max_experiments=MAX_EXPERIMENTS,
                    retain_neurons=RETAIN, forget_neurons=FORGET,
                    gpu_memory_utilization=GPU_MEM,
                )
                sp = SamplingParams(n=1, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
                sp.output_kind = RequestOutputKind.FINAL_ONLY
                prompts = make_prompts(N_PROMPTS * N_COMPLETIONS)
                eids = [1] * (N_PROMPTS * N_COMPLETIONS)
                for _ in range(N_WARMUP):
                    await mgr.generate(prompts, eids, sp)
                times = []
                for _ in range(N_TRIALS):
                    t0 = time.perf_counter()
                    results = await mgr.generate(prompts, eids, sp)
                    times.append(time.perf_counter() - t0)
                    assert len(results) == N_PROMPTS * N_COMPLETIONS
                engine.shutdown()
                del engine, mgr
                gc.collect()
                return times

            result_queue.put(asyncio.run(run()))

    except Exception as e:
        import traceback
        traceback.print_exc()
        result_queue.put(e)


def run_test(name, fn_name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_in_subprocess, args=(fn_name, q))
    p.start()
    p.join(timeout=300)
    if p.is_alive():
        p.kill()
        print("  TIMEOUT (300s)")
        return None
    result = q.get()
    if isinstance(result, Exception):
        print(f"  ERROR: {result}")
        return None
    times = result
    s = sorted(times)
    med = s[len(s) // 2]
    print(f"  Trials: {['%.3fs' % t for t in times]}")
    print(f"  Median: {med:.3f}s")
    return med


def main():
    total_seqs = N_PROMPTS * N_COMPLETIONS

    print(f"Benchmark: AsyncLLM overhead investigation")
    print(f"Model: {MODEL}")
    print(f"Adapters: retain={RETAIN}, forget={FORGET}")
    print(f"Workload: {N_PROMPTS} prompts x n={N_COMPLETIONS} = {total_seqs} seqs, max_tokens={MAX_TOKENS}")
    print(f"Trials: {N_TRIALS} (+ {N_WARMUP} warmup)")

    tests = [
        ("Sync LLM.generate (FINAL_ONLY, multiprocess)", "sync"),
        ("Sync LLM.generate (FINAL_ONLY, InprocClient)", "sync_inproc"),
        ("Async add_request (CUMULATIVE, default)", "async_cumulative"),
        ("Async add_request (FINAL_ONLY)", "async_final_only"),
        ("Async add_request (FINAL_ONLY, n=1 x 512)", "async_n1"),
    ]

    results = {}
    for name, fn_name in tests:
        med = run_test(name, fn_name)
        if med is not None:
            results[name] = med

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({N_PROMPTS}x n={N_COMPLETIONS} = {total_seqs} seqs)")
    print(f"{'='*60}")
    baseline = results.get("Sync LLM.generate (FINAL_ONLY, multiprocess)")
    for name, t in results.items():
        ratio = t / baseline if baseline else 0
        print(f"  {name:50s}  {t:.3f}s  ({ratio:.2f}x)")


if __name__ == "__main__":
    main()
