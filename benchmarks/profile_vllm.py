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


def export_sqlite(nsys_rep, output_dir):
    """Convert .nsys-rep to .sqlite."""
    sqlite_path = os.path.join(output_dir, "profile.sqlite")
    cmd = ["nsys", "export", "--type=sqlite", f"--output={sqlite_path}", nsys_rep]
    print(f"Exporting to SQLite...", flush=True)
    subprocess.run(cmd, capture_output=True)
    return sqlite_path


def extract_summary(sqlite_path, output_dir, args):
    """Extract GPU metrics summary and write tagged summary.txt."""
    import sqlite3
    import collections

    conn = sqlite3.connect(sqlite_path)

    lines = []
    lines.append("=" * 70)
    lines.append("vLLM Profile Summary")
    lines.append("=" * 70)
    lines.append(f"Model:          {args.model}")
    lines.append(f"N seqs:         {args.n_seqs}")
    lines.append(f"Max tokens:     {args.max_tokens}")
    lines.append(f"Prompt len:     {args.prompt_len}")
    lines.append(f"Num generations:{args.num_generations}")
    lines.append(f"Max num seqs:   {args.max_num_seqs}")
    lines.append(f"MLP adapters:   {args.mlp_adapters}")
    if args.mlp_adapters:
        lines.append(f"  retain:       {args.retain}")
        lines.append(f"  forget:       {args.forget}")
    lines.append(f"GPU mem util:   {args.gpu_mem}")
    lines.append(f"Duration:       {args.duration}s")
    lines.append("")

    # Duration from GPU metrics
    try:
        mn, mx = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM GPU_METRICS").fetchone()
        dur = (mx - mn) / 1e9
        lines.append(f"Profile duration (GPU metrics): {dur:.1f}s")
    except Exception:
        dur = 0
        lines.append("No GPU metrics found (need root for --gpu-metrics-devices)")

    # GPU utilization averages
    if dur > 0:
        tid = conn.execute("SELECT DISTINCT typeId FROM GPU_METRICS").fetchone()[0]
        # Skip first 10% as warmup, but no more than 30s
        warmup_ns = min(30_000_000_000, int(dur * 0.1 * 1_000_000_000))
        t_start = mn + warmup_ns
        metrics = {
            3: "SMs Active", 4: "SM Issue", 5: "Tensor Active",
            12: "Compute Warps in Flight", 15: "Unallocated Warps in Active SMs",
            18: "DRAM Read BW", 19: "DRAM Write BW",
        }
        warmup_s = warmup_ns / 1e9
        lines.append(f"\n--- GPU Averages (after {warmup_s:.1f}s warmup) ---")
        for mid, name in metrics.items():
            r = conn.execute(
                "SELECT AVG(value) FROM GPU_METRICS WHERE typeId=? AND metricId=? AND timestamp>=?",
                (tid, mid, t_start),
            ).fetchone()
            val = r[0] if r[0] else 0
            lines.append(f"  {name:>40}: {val:.1f}%")

        # Time series (10s bins)
        data = conn.execute(
            "SELECT timestamp, metricId, value FROM GPU_METRICS WHERE typeId=? AND metricId IN (3,5,12,18,19) ORDER BY timestamp",
            (tid,),
        ).fetchall()
        bins = collections.defaultdict(lambda: collections.defaultdict(list))
        for ts, mid, val in data:
            bucket = int((ts - mn) / 10_000_000_000) * 10
            bins[bucket][mid].append(val)

        lines.append(f"\n--- GPU Utilization Time Series (10s bins) ---")
        lines.append(f"{'Time':>8}  {'SMs Active':>11}  {'Tensor':>8}  {'Compute':>9}  {'DRAM Rd':>9}  {'DRAM Wr':>9}")
        for t in sorted(bins.keys()):
            b = bins[t]
            avg = lambda mid: sum(b[mid]) / len(b[mid]) if b[mid] else 0
            lines.append(f"{t:>6}s  {avg(3):>10.1f}%  {avg(5):>7.1f}%  {avg(12):>8.1f}%  {avg(18):>8.1f}%  {avg(19):>8.1f}%")

    # Top kernels
    try:
        rows = conn.execute("""
            SELECT demangledName, COUNT(*) as cnt, SUM(end-start) as total_ns, AVG(end-start) as avg_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            GROUP BY demangledName ORDER BY total_ns DESC LIMIT 15
        """).fetchall()
        total_kern = conn.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
        lines.append(f"\n--- Top 15 GPU Kernels (total kernel time: {total_kern/1e9:.1f}s) ---")
        for nameId, cnt, total_ns, avg_ns in rows:
            if isinstance(nameId, int):
                r = conn.execute("SELECT value FROM StringIds WHERE id=?", (nameId,)).fetchone()
                name = r[0] if r else str(nameId)
            else:
                name = nameId
            short = (name[:80] + "...") if len(name) > 80 else name
            pct = total_ns / total_kern * 100
            lines.append(f"  {pct:>5.1f}%  {cnt:>8}x  {total_ns/1e9:>7.2f}s  {avg_ns/1e3:>8.1f}us  {short}")
    except Exception as e:
        lines.append(f"\nNo kernel data: {e}")

    conn.close()

    summary = "\n".join(lines)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(summary)
    print(f"\nSummary written to {summary_path}")


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

    sqlite_path = export_sqlite(nsys_rep, output_dir)
    extract_summary(sqlite_path, output_dir, args)


if __name__ == "__main__":
    main()
