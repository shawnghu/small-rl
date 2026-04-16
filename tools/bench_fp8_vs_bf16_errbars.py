"""5 trials each of fp8_only and bf16_only on 235B non-thinking — for error bars."""
import argparse
import json
import os
import statistics as stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FP8_PROVIDERS = ["DeepInfra", "Novita", "SiliconFlow", "Parasail", "AtlasCloud"]
BF16_PROVIDERS = ["WandB", "Crusoe"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tools/judge_bench_dataset.jsonl")
    parser.add_argument("--judge_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge_model", default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--concurrent", type=int, default=1024)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f]
    reps = (args.n + len(samples) - 1) // len(samples)
    bench_prompts = ([s["prompt_text"] for s in samples] * reps)[:args.n]
    bench_completions = ([s["completion"] for s in samples] * reps)[:args.n]
    gt = ([s["gt_is_hack"] for s in samples] * reps)[:args.n]
    print(f"Benchmark: {args.n} samples ({sum(gt)} GT+ / {sum(not x for x in gt)} GT-)")
    print(f"Model: {args.judge_model}, trials={args.trials} per variant\n")

    from rh_detectors import llm_judge

    CONFIGS = [
        ("fp8_only",  {"provider": {"only": FP8_PROVIDERS, "sort": "throughput",
                                    "allow_fallbacks": False}}),
        ("bf16_only", {"provider": {"only": BF16_PROVIDERS, "allow_fallbacks": True}}),
    ]

    def run_once(label, extra_body, trial):
        t0 = time.perf_counter()
        try:
            pred = llm_judge(
                bench_completions, prompts=bench_prompts,
                judge_url=args.judge_url, judge_model=args.judge_model,
                judge_extra_body=extra_body,
                concurrent=args.concurrent, max_tokens=args.max_tokens,
                temperature=args.temperature, require_thinking=False,
            )
            dt = time.perf_counter() - t0
            tp = sum(1 for p, g in zip(pred, gt) if p and g)
            fp = sum(1 for p, g in zip(pred, gt) if p and not g)
            fn = sum(1 for p, g in zip(pred, gt) if not p and g)
            tn = sum(1 for p, g in zip(pred, gt) if not p and not g)
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            return {"variant": label, "trial": trial, "wall_s": round(dt, 2),
                    "throughput_req_s": round(args.n / dt, 2),
                    "p": round(prec, 4), "r": round(rec, 4), "f1": round(f1, 4)}
        except Exception as e:
            dt = time.perf_counter() - t0
            return {"variant": label, "trial": trial, "wall_s": round(dt, 2),
                    "error": str(e)[:150]}

    jobs = [(label, extra, t) for label, extra in CONFIGS for t in range(args.trials)]
    print(f"Running {len(jobs)} trials SEQUENTIALLY to isolate each from rate-limit interference...\n")
    results = []
    for label, extra, t in jobs:
        r = run_once(label, extra, t)
        results.append(r)
        if "error" in r:
            print(f"[{r['variant']} t{r['trial']}] FAIL  err: {r['error'][:80]}", flush=True)
        else:
            print(f"[{r['variant']:<10} t{r['trial']}] wall={r['wall_s']:>5.1f}s  "
                  f"thr={r['throughput_req_s']:>6.2f}  "
                  f"P={r['p']:.3f}  R={r['r']:.3f}  F1={r['f1']:.3f}", flush=True)

    def stats(values):
        return {"mean": round(stat.mean(values), 4),
                "stdev": round(stat.stdev(values) if len(values) > 1 else 0, 4),
                "min": round(min(values), 4), "max": round(max(values), 4)}

    print("\n=== Summary (mean ± stdev over trials) ===")
    print(f"{'variant':<12} {'req/s':>18} {'P':>20} {'R':>20} {'F1':>20}")
    summary_records = []
    for label, _ in CONFIGS:
        good = [r for r in results if r["variant"] == label and "error" not in r]
        if not good:
            print(f"{label:<12}  all trials failed")
            continue
        thr = stats([r["throughput_req_s"] for r in good])
        p = stats([r["p"] for r in good])
        r_ = stats([r["r"] for r in good])
        f1 = stats([r["f1"] for r in good])
        print(f"{label:<12} {thr['mean']:>8.2f}±{thr['stdev']:>5.2f} "
              f"{p['mean']:>9.3f}±{p['stdev']:>5.3f} "
              f"{r_['mean']:>9.3f}±{r_['stdev']:>5.3f} "
              f"{f1['mean']:>9.3f}±{f1['stdev']:>5.3f}   "
              f"(n={len(good)} of {args.trials})")
        summary_records.append({"variant": label, "n_trials": len(good),
                                "throughput": thr, "precision": p,
                                "recall": r_, "f1": f1})

    out_path = "tools/bench_fp8_vs_bf16_errbars_results.json"
    with open(out_path, "w") as f:
        json.dump({"individual": results, "summary": summary_records}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
