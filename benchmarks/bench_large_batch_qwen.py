"""Profile vLLM throughput at large batch size with Qwen3.5-9B (no adapters)."""
import asyncio
import os
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

MODEL = "Qwen/Qwen3.5-0.6B"  # placeholder, overridden below
GPU_MEM = 0.9
N_SEQS = 512
MAX_TOKENS = 16


async def run(model, n_seqs, max_tokens):
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    tok = AutoTokenizer.from_pretrained(model)
    pid = tok.encode("Once upon a time", add_special_tokens=False)

    engine_args = AsyncEngineArgs(
        model=model,
        gpu_memory_utilization=GPU_MEM,
        dtype="auto",
        max_model_len=512,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sp = SamplingParams(n=1, temperature=1.0, max_tokens=max_tokens)
    sp.output_kind = RequestOutputKind.FINAL_ONLY

    # warmup
    print(f"Warmup: 32 seqs, max_tokens={max_tokens}")
    warmup_tasks = []
    for i in range(32):
        gen = engine.generate(TokensPrompt(prompt_token_ids=list(pid)), sp, request_id=f"warmup-{i}")
        warmup_tasks.append(gen)
    for gen in warmup_tasks:
        async for _ in gen:
            pass

    # main run
    print(f"Running: model={model}, n_seqs={n_seqs}, max_tokens={max_tokens}")
    t0 = time.perf_counter()
    tasks = []
    for i in range(n_seqs):
        gen = engine.generate(TokensPrompt(prompt_token_ids=list(pid)), sp, request_id=f"run-{i}")
        tasks.append(gen)
    results = []
    for gen in tasks:
        async for out in gen:
            pass
        results.append(out)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(r.outputs[0].token_ids) for r in results)
    print(f"  {elapsed:.2f}s, {len(results)} seqs, {total_tokens} tokens, {total_tokens/elapsed:.0f} tok/s")

    engine.shutdown()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--n_seqs", type=int, default=N_SEQS)
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    args = p.parse_args()
    asyncio.run(run(args.model, args.n_seqs, args.max_tokens))
