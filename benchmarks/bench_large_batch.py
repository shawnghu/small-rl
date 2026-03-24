"""Profile vLLM+MLP adapter throughput at large batch size."""
import asyncio
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
GPU_MEM = 0.8
MAX_EXPERIMENTS = 2
RETAIN = 32
FORGET = 32
N_SEQS = 16384
MAX_TOKENS = 256


async def run(n_seqs, max_tokens):
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm_mlp_adapter import create_async_engine

    tok = AutoTokenizer.from_pretrained(MODEL)
    pid = tok.encode("Once upon a time", add_special_tokens=False)

    engine, mgr = await create_async_engine(
        model_name=MODEL, max_experiments=MAX_EXPERIMENTS,
        retain_neurons=RETAIN, forget_neurons=FORGET,
        gpu_memory_utilization=GPU_MEM,
    )
    sp = SamplingParams(n=1, temperature=1.0, max_tokens=max_tokens)
    sp.output_kind = RequestOutputKind.FINAL_ONLY
    prompts = [TokensPrompt(prompt_token_ids=list(pid)) for _ in range(n_seqs)]
    eids = [1] * n_seqs

    print(f"Warmup: n_seqs=256, max_tokens={max_tokens}")
    await mgr.generate(prompts[:256], eids[:256], sp)

    print(f"Running: n_seqs={n_seqs}, max_tokens={max_tokens}")
    t0 = time.perf_counter()
    results = await mgr.generate(prompts, eids, sp)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(r.outputs[0].token_ids) for r in results)
    print(f"  {elapsed:.2f}s, {len(results)} seqs, {total_tokens} tokens, {total_tokens/elapsed:.0f} tok/s")

    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(run(N_SEQS, MAX_TOKENS))
