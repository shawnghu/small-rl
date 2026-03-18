"""Benchmark: concurrent vs sequential async batches.

5 callers each submit 32 prompts × n=16 completions (512 seqs each, 2560 total).

  Sequential async: 5 generate() calls one after another
  Concurrent async: asyncio.gather(5 generate() calls) — engine sees all 2560 at once
  Sequential sync:  same with sync LLM (baseline)
"""
import asyncio
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import SamplingParams
from vllm_mlp_adapter import create_engine, create_async_engine

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_CALLERS = 5
N_PROMPTS = 32
N_COMPLETIONS = 16   # n per prompt
MAX_TOKENS = 16
GPU_MEM = 0.1

PROMPTS = ["Once upon a time"] * N_PROMPTS
EIDS = [1] * N_PROMPTS
SYNC_PARAMS = SamplingParams(n=N_COMPLETIONS, temperature=1.0, max_tokens=MAX_TOKENS,
                             ignore_eos=True)
ASYNC_PARAMS = SamplingParams(n=N_COMPLETIONS, temperature=1.0, max_tokens=MAX_TOKENS,
                              ignore_eos=True)


def bench_sync():
    from vllm import SamplingParams
    llm, mgr = create_engine(MODEL, gpu_memory_utilization=GPU_MEM)

    # warm up
    mgr.generate(PROMPTS[:1], [1], SamplingParams(n=1, temperature=1.0, max_tokens=4))

    times = []
    for _ in range(N_CALLERS):
        t0 = time.perf_counter()
        outs = mgr.generate(PROMPTS, EIDS, SYNC_PARAMS)
        times.append(time.perf_counter() - t0)
        total = sum(len(o.outputs) for o in outs)
        assert total == N_PROMPTS * N_COMPLETIONS, f"got {total}"

    total_t = sum(times)
    print(f"  Sequential sync  ({N_CALLERS} calls):  "
          f"total={total_t:.3f}s  per-call=[{', '.join(f'{t:.3f}' for t in times)}]")


async def bench_async():
    engine, mgr = await create_async_engine(
        MODEL, gpu_memory_utilization=GPU_MEM,
        max_num_seqs=N_CALLERS * N_PROMPTS * N_COMPLETIONS + 256,
    )

    # warm up
    await mgr.generate(PROMPTS[:1], [1],
                       SamplingParams(n=1, temperature=1.0, max_tokens=4))

    # sequential
    seq_times = []
    for _ in range(N_CALLERS):
        t0 = time.perf_counter()
        outs = await mgr.generate(PROMPTS, EIDS, ASYNC_PARAMS)
        seq_times.append(time.perf_counter() - t0)
        total = sum(len(o.outputs) for o in outs)
        assert total == N_PROMPTS * N_COMPLETIONS, f"got {total}"

    seq_total = sum(seq_times)
    print(f"  Sequential async ({N_CALLERS} calls):  "
          f"total={seq_total:.3f}s  per-call=[{', '.join(f'{t:.3f}' for t in seq_times)}]")

    # concurrent
    t0 = time.perf_counter()
    all_outs = await asyncio.gather(
        *(mgr.generate(PROMPTS, EIDS, ASYNC_PARAMS) for _ in range(N_CALLERS))
    )
    conc_total = time.perf_counter() - t0
    for outs in all_outs:
        total = sum(len(o.outputs) for o in outs)
        assert total == N_PROMPTS * N_COMPLETIONS, f"got {total}"

    print(f"  Concurrent async ({N_CALLERS} callers): total={conc_total:.3f}s  "
          f"(vs sequential: {seq_total / conc_total:.1f}x faster)")


if __name__ == "__main__":
    print(f"\n{N_CALLERS} callers × {N_PROMPTS} prompts × n={N_COMPLETIONS} "
          f"= {N_CALLERS * N_PROMPTS * N_COMPLETIONS} total seqs, max_tokens={MAX_TOKENS}\n")

    print("--- Sync (sequential baseline) ---")
    bench_sync()

    print("\n--- Async ---")
    asyncio.run(bench_async())
