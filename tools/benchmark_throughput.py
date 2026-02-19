"""Benchmark concurrent training throughput.

Launches N jobs simultaneously, polls logs until each job has logged at least
MIN_STEPS step_time entries (skipping the first WARMUP_STEPS), then kills jobs
and reports. No need to run to completion.

Usage examples:
    python tools/benchmark_throughput.py --concurrency 1 4 8 12 16 20 24
    python tools/benchmark_throughput.py --model HuggingFaceTB/SmolLM-135M \
        --concurrency 1 2 4 6 8 10 12 --batch_sizes 32 64 128
"""

import argparse
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "SimpleStories/SimpleStories-1.25M"

WARMUP_STEPS = 2   # skip first N step_time entries (JIT warmup)
MIN_STEPS = 3      # collect at least this many stable steps per job
POLL_INTERVAL = 2  # seconds between log polls
MAX_WAIT = 300     # hard timeout per trial (seconds)

STEP_TIME_RE = re.compile(r"'step_time': '([0-9.]+)'")


def parse_step_times(log_path: Path) -> list[float]:
    try:
        return [float(m) for m in STEP_TIME_RE.findall(log_path.read_text())]
    except Exception:
        return []


def launch_job(i: int, run_dir: Path, args: argparse.Namespace, bs: int) -> subprocess.Popen:
    cmd = [
        "uv", "run", "python", str(REPO_ROOT / "train.py"),
        "--reward", args.reward,
        "--lora_config", args.lora_config,
        "--batch_size", str(bs),
        "--lr", str(args.lr),
        "--beta", str(args.beta),
        "--num_generations", str(args.num_generations),
        "--max_steps", "9999",
        "--logging_steps", "1",
        "--no_wandb",
        "--verbose",
        "--save_steps", "999999",
        "--output_dir", str(run_dir / f"job_{i}"),
        "--seed", str(args.seed + i),
    ]
    if args.model != DEFAULT_MODEL:
        cmd += ["--model", args.model]
    log_file = open(run_dir / f"job_{i}.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    proc._log = run_dir / f"job_{i}.log"
    proc._log_file = log_file
    return proc


def run_trial(n: int, bs: int, args: argparse.Namespace, tmp_base: Path) -> dict | None:
    run_dir = tmp_base / f"n{n}_bs{bs}"
    run_dir.mkdir(parents=True, exist_ok=True)

    procs = [launch_job(i, run_dir, args, bs) for i in range(n)]
    t_start = time.monotonic()

    collected = [[] for _ in range(n)]
    deadline = t_start + MAX_WAIT

    try:
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)
            for i, p in enumerate(procs):
                if p.poll() is not None:
                    continue  # already done (or crashed)
                times = parse_step_times(p._log)
                collected[i] = times[WARMUP_STEPS:]  # skip warmup

            ready = sum(1 for c in collected if len(c) >= MIN_STEPS)
            if ready == n:
                break
            # If most jobs have enough data and remainder are failing/slow, proceed
            alive = sum(1 for p in procs if p.poll() is None)
            if ready >= max(1, n - 2) and alive < n:
                break
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
            p._log_file.close()
        # Delete model checkpoint dirs, keep logs
        for i in range(n):
            d = run_dir / f"job_{i}"
            if d.exists():
                shutil.rmtree(d)

    all_times = [t for c in collected for t in c]
    if not all_times:
        return None

    avg_step_time = sum(all_times) / len(all_times)
    agg = n / avg_step_time
    return {
        "n": n, "bs": bs,
        "avg_step_time": avg_step_time,
        "agg_steps_per_s": agg,
        "samples_per_s": bs * agg,
        "n_times": len(all_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 12, 16, 20, 24])
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lora_config", default="r32")
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--reward", default="happy_binary")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tmp_dir", default="/tmp/throughput_bench")
    args = parser.parse_args()

    batch_sizes = args.batch_sizes or [args.batch_size]
    tmp_base = Path(args.tmp_dir)
    tmp_base.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.split("/")[-1]
    print(f"model={model_tag}  lora={args.lora_config}  bs={batch_sizes}  n={args.concurrency}")
    print(f"(sampling {MIN_STEPS} steps/job after {WARMUP_STEPS} warmup, then killing)\n")

    results = []
    for bs in batch_sizes:
        print(f"=== bs={bs} ===")
        baseline = None
        for n in args.concurrency:
            print(f"  n={n} ...", end=" ", flush=True)
            r = run_trial(n, bs, args, tmp_base)
            if r is None:
                print("FAILED")
                continue
            results.append(r)
            if baseline is None:
                baseline = r["agg_steps_per_s"]
            eff = r["agg_steps_per_s"] / baseline
            print(f"step={r['avg_step_time']:.2f}s  agg={r['agg_steps_per_s']:.1f}/s  "
                  f"samples={r['samples_per_s']:.0f}/s  eff={eff:.2f}x")
        print()

    print(f"{'bs':>6} {'n':>4} {'step_time':>10} {'steps/s':>8} {'samples/s':>10}")
    print("-" * 44)
    for r in results:
        print(f"{r['bs']:>6} {r['n']:>4} {r['avg_step_time']:>9.2f}s "
              f"{r['agg_steps_per_s']:>8.1f} {r['samples_per_s']:>10.0f}")


if __name__ == "__main__":
    main()
