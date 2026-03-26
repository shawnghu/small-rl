#!/usr/bin/env python3
"""
Profile vLLM generation throughput under nsys and extract GPU metrics.

Usage (as root for GPU metrics):
    python benchmarks/profile_vllm.py --model Qwen/Qwen3-8B --n_seqs 512 --max_tokens 16 --gpu 0

    python benchmarks/profile_vllm.py --model HuggingFaceTB/SmolLM2-135M-Instruct --n_seqs 512 --max_tokens 256 --mlp_adapters

Prompts are random token sequences of --prompt_len (default 16), duplicated --num_generations (default 16) times each to simulate GRPO rollouts.

Output goes to benchmarks/profiles/{tag}/ with:
    - profile.nsys-rep
    - profile.sqlite
    - summary.txt (tagged with parameters)
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsys_utils import export_sqlite, extract_summary


def make_tag(args):
    model_short = args.model.split("/")[-1]
    parts = [model_short, f"n{args.n_seqs}", f"t{args.max_tokens}", f"p{args.prompt_len}", f"g{args.num_generations}", f"mns{args.max_num_seqs}"]
    if args.mlp_adapters:
        parts.append(f"mlp_r{args.retain}_f{args.forget}")
    parts.append(f"dur{args.duration}")
    return "_".join(parts)


def write_vllm_worker(path, args):
    """Write a temporary Python script that runs vLLM generation in a loop."""
    mlp_section = ""
    if args.mlp_adapters:
        mlp_section = f"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vllm_mlp_adapter import create_async_engine
    engine, mgr = await create_async_engine(
        model_name="{args.model}", max_experiments=2,
        retain_neurons={args.retain}, forget_neurons={args.forget},
        gpu_memory_utilization={args.gpu_mem},
    )
"""
    else:
        mlp_section = f"""
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    engine_args = AsyncEngineArgs(
        model="{args.model}",
        gpu_memory_utilization={args.gpu_mem},
        dtype="auto",
        max_model_len=512,
        max_num_seqs={args.max_num_seqs},
        max_num_batched_tokens={args.max_num_batched_tokens},
        disable_log_stats=False,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
"""

    if args.mlp_adapters:
        generate_section = f"""
    eids = [1] * n_seqs
    # warmup
    print(f"Warmup: 32 seqs", flush=True)
    await mgr.generate(all_prompts[:32], eids[:32], sp)
    print(f"Running: n_seqs={{n_seqs}}, max_tokens={{max_tokens}}", flush=True)
    t0 = time.perf_counter()
    results = await mgr.generate(all_prompts, eids, sp)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(r.outputs[0].token_ids) for r in results)
    print(f"  {{elapsed:.2f}}s, {{len(results)}} seqs, {{total_tokens}} tokens, {{total_tokens/elapsed:.0f}} tok/s", flush=True)
    engine.shutdown()
"""
    else:
        generate_section = f"""
    # warmup
    print(f"Warmup: 32 seqs", flush=True)
    warmup_tasks = []
    for i in range(32):
        gen = engine.generate(all_prompts[i], sp, request_id=f"warmup-{{i}}")
        warmup_tasks.append(gen)
    for gen in warmup_tasks:
        async for _ in gen:
            pass
    print(f"Running: n_seqs={{n_seqs}}, max_tokens={{max_tokens}}", flush=True)
    t0 = time.perf_counter()
    tasks = []
    for i in range(n_seqs):
        gen = engine.generate(all_prompts[i], sp, request_id=f"run-{{i}}")
        tasks.append(gen)
    results = []
    for gen in tasks:
        async for out in gen:
            pass
        results.append(out)
    elapsed = time.perf_counter() - t0
    total_tokens = sum(len(r.outputs[0].token_ids) for r in results)
    print(f"  {{elapsed:.2f}}s, {{len(results)}} seqs, {{total_tokens}} tokens, {{total_tokens/elapsed:.0f}} tok/s", flush=True)
    engine.shutdown()
"""

    script = f'''#!/usr/bin/env python3
"""Auto-generated vLLM benchmark worker."""
import asyncio
import os
import sys
import time
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

async def run(n_seqs, max_tokens):
    from transformers import AutoTokenizer
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
{mlp_section}
    import random
    tok = AutoTokenizer.from_pretrained("{args.model}")
    vocab_size = tok.vocab_size
    n_unique = n_seqs // {args.num_generations}
    assert n_seqs % {args.num_generations} == 0, f"n_seqs must be divisible by num_generations ({args.num_generations})"
    rng = random.Random(42)
    unique_prompts = [[rng.randint(0, vocab_size - 1) for _ in range({args.prompt_len})] for _ in range(n_unique)]
    all_prompts = [TokensPrompt(prompt_token_ids=list(p)) for p in unique_prompts for _ in range({args.num_generations})]
    sp = SamplingParams(n=1, temperature=1.0, max_tokens=max_tokens)
    sp.output_kind = RequestOutputKind.FINAL_ONLY
{generate_section}

if __name__ == "__main__":
    asyncio.run(run({args.n_seqs}, {args.max_tokens}))
'''
    with open(path, "w") as f:
        f.write(script)


def run_nsys(worker_path, output_dir, args):
    """Run nsys profile on the worker script."""
    nsys_rep = os.path.join(output_dir, "profile.nsys-rep")
    nsys_base = os.path.join(output_dir, "profile")

    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--gpu-metrics-devices=0",
        "--gpu-metrics-frequency=10000",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-event-trace=false",
        "--trace-fork-before-exec=true",
        "--wait=primary",
        f"--duration={args.duration}",
        f"-o", nsys_base,
        sys.executable, "-u", worker_path,
    ]

    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"nsys exited with code {result.returncode}", file=sys.stderr)
    return nsys_rep


def _make_vllm_header(args):
    """Build summary header lines for a vLLM profile."""
    lines = [
        "vLLM Profile Summary",
        "",
        f"Model:          {args.model}",
        f"N seqs:         {args.n_seqs}",
        f"Max tokens:     {args.max_tokens}",
        f"Prompt len:     {args.prompt_len}",
        f"Num generations:{args.num_generations}",
        f"Max num seqs:   {args.max_num_seqs}",
        f"MLP adapters:   {args.mlp_adapters}",
    ]
    if args.mlp_adapters:
        lines.append(f"  retain:       {args.retain}")
        lines.append(f"  forget:       {args.forget}")
    lines.append(f"GPU mem util:   {args.gpu_mem}")
    lines.append(f"Duration:       {args.duration}s")
    lines.append("")
    return lines


def main():
    p = argparse.ArgumentParser(description="Profile vLLM generation under nsys")
    p.add_argument("--model", required=True)
    p.add_argument("--n_seqs", type=int, default=512)
    p.add_argument("--max_tokens", type=int, default=16)
    p.add_argument("--delay", type=int, default=60, help="seconds before nsys starts collecting")
    p.add_argument("--duration", type=int, default=20, help="seconds of nsys collection after delay")
    p.add_argument("--mlp_adapters", action="store_true")
    p.add_argument("--retain", type=int, default=32)
    p.add_argument("--forget", type=int, default=32)
    p.add_argument("--gpu_mem", type=float, default=0.9)
    p.add_argument("--gpu", type=int, default=0, help="physical GPU index")
    p.add_argument("--max_num_seqs", type=int, default=4096)
    p.add_argument("--max_num_batched_tokens", type=int, default=65536)
    p.add_argument("--prompt_len", type=int, default=16, help="length of random token prompts")
    p.add_argument("--num_generations", type=int, default=16, help="duplicate each unique prompt this many times")
    p.add_argument("--output_dir", default="benchmarks/profiles")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    tag = make_tag(args)
    output_dir = os.path.join(args.output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    worker_path = os.path.join(output_dir, "_worker.py")
    write_vllm_worker(worker_path, args)

    nsys_rep = run_nsys(worker_path, output_dir, args)

    if not os.path.exists(nsys_rep):
        print(f"ERROR: {nsys_rep} not found — nsys may have failed", file=sys.stderr)
        sys.exit(1)

    sqlite_path = export_sqlite(nsys_rep)
    summary_path = os.path.join(output_dir, "summary.txt")
    extract_summary(sqlite_path, _make_vllm_header(args), summary_path)


if __name__ == "__main__":
    main()
