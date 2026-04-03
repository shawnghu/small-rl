"""Profile vLLM throughput at large batch size (no adapters)."""
import asyncio
import os
import random
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

GPU_MEM = 0.9
N_SEQS = 512
MAX_TOKENS = 16
PROMPT_LEN = 200


async def run(model, n_seqs, max_tokens, prompt_len):
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    tok = AutoTokenizer.from_pretrained(model)
    vocab_size = tok.vocab_size

    max_model_len = prompt_len + max_tokens
    engine_args = AsyncEngineArgs(
        model=model,
        gpu_memory_utilization=GPU_MEM,
        dtype="auto",
        max_model_len=max_model_len,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    rng = random.Random(42)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randint(0, vocab_size - 1) for _ in range(prompt_len)]) for _ in range(n_seqs)]

    sp = SamplingParams(n=1, temperature=1.0, max_tokens=max_tokens)
    sp.output_kind = RequestOutputKind.FINAL_ONLY

    # warmup
    print(f"Warmup: 32 seqs, prompt_len={prompt_len}, max_tokens={max_tokens}")
    warmup_tasks = []
    for i in range(32):
        gen = engine.generate(prompts[i % len(prompts)], sp, request_id=f"warmup-{i}")
        warmup_tasks.append(gen)
    for gen in warmup_tasks:
        async for _ in gen:
            pass

    # main run
    print(f"Running: model={model}, n_seqs={n_seqs}, prompt_len={prompt_len}, max_tokens={max_tokens}")
    t0 = time.perf_counter()
    tasks = []
    for i in range(n_seqs):
        gen = engine.generate(prompts[i], sp, request_id=f"run-{i}")
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
    p.add_argument("--prompt_len", type=int, default=PROMPT_LEN)
    args = p.parse_args()
    asyncio.run(run(args.model, args.n_seqs, args.max_tokens, args.prompt_len))
