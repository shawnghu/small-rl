"""Profile vLLM throughput at large batch size (no adapters).

Uses synchronous LLM.generate() API to ensure proper batching.
The async API can serialize requests through Python's event loop,
giving misleadingly low throughput numbers.
"""
import os
import random
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

GPU_MEM = 0.9
N_SEQS = 2048
MAX_TOKENS = 64
PROMPT_LEN = 200


def run(model, n_seqs, max_tokens, prompt_len):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model)
    vocab_size = tok.vocab_size

    max_model_len = prompt_len + max_tokens
    llm = LLM(
        model=model,
        gpu_memory_utilization=GPU_MEM,
        dtype="auto",
        max_model_len=max_model_len,
        max_num_seqs=2048,
    )

    rng = random.Random(42)
    prompts = [{"prompt_token_ids": [rng.randint(0, vocab_size - 1) for _ in range(prompt_len)]} for _ in range(n_seqs)]

    sp = SamplingParams(n=1, temperature=1.0, max_tokens=max_tokens)

    # warmup
    print(f"Warmup: 64 seqs, prompt_len={prompt_len}, max_tokens={max_tokens}")
    llm.generate(prompts[:64], sp)

    # main run
    print(f"Running: model={model}, n_seqs={n_seqs}, prompt_len={prompt_len}, max_tokens={max_tokens}")
    t0 = time.perf_counter()
    results = llm.generate(prompts, sp)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(r.outputs[0].token_ids) for r in results)
    full = sum(1 for r in results if len(r.outputs[0].token_ids) >= max_tokens)
    print(f"  {elapsed:.2f}s, {len(results)} seqs, {total_tokens} tokens, {total_tokens/elapsed:.0f} tok/s")
    print(f"  Seqs hitting max_tokens: {full}/{len(results)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--n_seqs", type=int, default=N_SEQS)
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--prompt_len", type=int, default=PROMPT_LEN)
    args = p.parse_args()
    run(args.model, args.n_seqs, args.max_tokens, args.prompt_len)
