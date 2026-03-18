"""cProfile the async generate() call to find the bottleneck."""
import asyncio
import cProfile
import os
import pstats
import sys

os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
GPU_MEM = 0.1
MAX_EXPERIMENTS = 6
RETAIN = 32
FORGET = 32
N_PROMPTS = 32
N_COMPLETIONS = 16
MAX_TOKENS = 16


async def main():
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    from vllm_mlp_adapter import create_async_engine

    tok = AutoTokenizer.from_pretrained(MODEL)
    pid = tok.encode("Once upon a time", add_special_tokens=False)
    prompts = [TokensPrompt(prompt_token_ids=list(pid)) for _ in range(N_PROMPTS)]
    eids = [1] * N_PROMPTS

    engine, mgr = await create_async_engine(
        model_name=MODEL, max_experiments=MAX_EXPERIMENTS,
        retain_neurons=RETAIN, forget_neurons=FORGET,
        gpu_memory_utilization=GPU_MEM,
    )
    sp = SamplingParams(n=N_COMPLETIONS, temperature=1.0, max_tokens=MAX_TOKENS)
    sp.output_kind = RequestOutputKind.FINAL_ONLY

    # warmup (not profiled)
    await mgr.generate(prompts, eids, sp)

    # profiled run
    pr = cProfile.Profile()
    pr.enable()
    results = await mgr.generate(prompts, eids, sp)
    pr.disable()

    total = sum(len(r.outputs) for r in results)
    assert total == N_PROMPTS * N_COMPLETIONS, f"got {total}"

    ps = pstats.Stats(pr, stream=sys.stdout)
    ps.sort_stats("cumulative")
    ps.print_stats(40)

    engine.shutdown()


asyncio.run(main())
