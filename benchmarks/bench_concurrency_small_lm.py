"""Launch N parallel bench_training_step.py copies on a single GPU under MPS and
measure aggregate experiment-steps/sec for the liger+dynamic+gc=on config.

Usage:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python bench_concurrency_gpu6.py \
      --concurrencies 1,2,4,6,8,12,16 --steps 30 --warmup 15
"""

import argparse
import json
import os
import subprocess
import time


def run_cell(n, steps, warmup, results_dir):
    procs = []
    result_files = []
    os.makedirs(results_dir, exist_ok=True)
    env = os.environ.copy()
    t0 = time.perf_counter()
    for i in range(n):
        rf = os.path.join(results_dir, f"conc{n}_p{i}.json")
        result_files.append(rf)
        cmd = [
            ".venv/bin/python", "bench_training_step.py",
            "--batch", "synthetic_512_10to50_c32.pt",
            "--sweep_config", "sweeps/basic_coherence_sweep.py",
            "--run_index", "0",
            "--bench_steps", str(steps),
            "--warmup_steps", str(warmup),
            "--gradient_checkpointing", "true",
            "--results", rf,
        ]
        log = open(os.path.join(results_dir, f"conc{n}_p{i}.log"), "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
    for p in procs:
        p.wait()
    wall = time.perf_counter() - t0

    per_proc_means = []
    per_proc_medians = []
    per_proc_p95s = []
    peaks = []
    for rf in result_files:
        if not os.path.exists(rf):
            continue
        d = json.load(open(rf))
        st = d["step_times"]
        per_proc_means.append(st["mean"])
        per_proc_medians.append(st["median"])
        per_proc_p95s.append(st["p95"])
        peaks.append(d["memory"]["peak_update_gb"])

    ok = len(per_proc_means)
    if ok == 0:
        return {"n": n, "ok": 0, "wall": wall}
    avg_mean = sum(per_proc_means) / ok
    avg_median = sum(per_proc_medians) / ok
    worst_p95 = max(per_proc_p95s)
    agg_steps_per_sec = ok / avg_mean
    return {
        "n": n,
        "ok": ok,
        "wall": wall,
        "per_proc_mean_ms": avg_mean * 1000,
        "per_proc_median_ms": avg_median * 1000,
        "per_proc_worst_p95_ms": worst_p95 * 1000,
        "agg_steps_per_sec": agg_steps_per_sec,
        "per_proc_peak_gb_max": max(peaks),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrencies", default="1,2,4,6,8,12,16")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--results_dir", default="bench_results/concurrency")
    args = p.parse_args()
    ns = [int(x) for x in args.concurrencies.split(",")]
    results = []
    for n in ns:
        print(f"\n=== concurrency {n} ===", flush=True)
        r = run_cell(n, args.steps, args.warmup, args.results_dir)
        print(r, flush=True)
        results.append(r)

    print("\n\n=== SUMMARY ===")
    print(f"{'n':>4} {'ok':>4} {'wall':>8} {'p_mean_ms':>10} {'p_med_ms':>10} {'worst_p95':>10} {'agg_steps/s':>12} {'peakGB':>8}")
    for r in results:
        if r.get("ok", 0) == 0:
            print(f"{r['n']:>4} {'FAIL':>4}")
            continue
        print(f"{r['n']:>4} {r['ok']:>4} {r['wall']:8.1f} "
              f"{r['per_proc_mean_ms']:10.1f} {r['per_proc_median_ms']:10.1f} "
              f"{r['per_proc_worst_p95_ms']:10.1f} {r['agg_steps_per_sec']:12.2f} "
              f"{r['per_proc_peak_gb_max']:8.2f}")


if __name__ == "__main__":
    main()
