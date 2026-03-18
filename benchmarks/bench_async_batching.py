"""Test if async engine batches all requests in one step vs sync."""
import asyncio
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
GPU_MEM = 0.1
MAX_EXPERIMENTS = 6
RETAIN = 32
FORGET = 32


def run_sync(n_seqs, max_tokens):
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt
    from vllm_mlp_adapter import create_engine

    tok = AutoTokenizer.from_pretrained(MODEL)
    pid = tok.encode("Once upon a time", add_special_tokens=False)

    llm, mgr = create_engine(model_name=MODEL, max_experiments=MAX_EXPERIMENTS,
                              retain_neurons=RETAIN, forget_neurons=FORGET,
                              gpu_memory_utilization=GPU_MEM)
    sp = SamplingParams(n=1, temperature=1.0, max_tokens=max_tokens)
    prompts = [TokensPrompt(prompt_token_ids=list(pid)) for _ in range(n_seqs)]
    eids = [1] * n_seqs

    # warmup
    mgr.generate(prompts, eids, sp)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        results = mgr.generate(prompts, eids, sp)
        times.append(time.perf_counter() - t0)
        assert len(results) == n_seqs, f"got {len(results)}"

    med = sorted(times)[1]
    print(f"  Sync  n={n_seqs} max_tokens={max_tokens}: median={med:.3f}s  {[f'{t:.3f}' for t in times]}")
    del llm, mgr
    return med


async def run_async(n_seqs, max_tokens):
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm_mlp_adapter import create_async_engine

    tok = AutoTokenizer.from_pretrained(MODEL)
    pid = tok.encode("Once upon a time", add_special_tokens=False)

    engine, mgr = await create_async_engine(model_name=MODEL, max_experiments=MAX_EXPERIMENTS,
                                             retain_neurons=RETAIN, forget_neurons=FORGET,
                                             gpu_memory_utilization=GPU_MEM)
    sp = SamplingParams(n=1, temperature=1.0, max_tokens=max_tokens)
    sp.output_kind = RequestOutputKind.FINAL_ONLY
    prompts = [TokensPrompt(prompt_token_ids=list(pid)) for _ in range(n_seqs)]
    eids = [1] * n_seqs

    # warmup
    await mgr.generate(prompts, eids, sp)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        results = await mgr.generate(prompts, eids, sp)
        times.append(time.perf_counter() - t0)
        assert len(results) == n_seqs, f"got {len(results)}"

    med = sorted(times)[1]
    print(f"  Async n={n_seqs} max_tokens={max_tokens}: median={med:.3f}s  {[f'{t:.3f}' for t in times]}")
    engine.shutdown()
    del engine, mgr
    return med


import multiprocessing as mp


def _sync_worker(n_seqs, max_tokens, q):
    import gc
    try:
        t = run_sync(n_seqs, max_tokens)
        q.put(t)
    except Exception as e:
        import traceback; traceback.print_exc()
        q.put(e)
    gc.collect()


def _async_worker(n_seqs, max_tokens, q):
    import gc
    try:
        t = asyncio.run(run_async(n_seqs, max_tokens))
        q.put(t)
    except Exception as e:
        import traceback; traceback.print_exc()
        q.put(e)
    gc.collect()


def run_in_subprocess(fn, *args):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=fn, args=(*args, q))
    p.start()
    p.join(timeout=300)
    if p.is_alive():
        p.kill()
        return None
    r = q.get()
    if isinstance(r, Exception):
        return None
    return r


if __name__ == "__main__":
    print("Testing batching: does async process all requests in one batch?\n")
    for n_seqs, max_tokens in [(32, 16), (128, 16), (512, 16), (32, 64)]:
        print(f"\n--- n_seqs={n_seqs}, max_tokens={max_tokens} ---")
        run_in_subprocess(_sync_worker, n_seqs, max_tokens)
        run_in_subprocess(_async_worker, n_seqs, max_tokens)
