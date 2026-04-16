"""Benchmark 235B non-thinking across quantization filters.

Variants (all use full 'reward_hacking_binary' prompt, sort by throughput
within each allowed set):
  default       : no filter (likely routes to fp8 providers)
  no_fp8        : ignore fp8 providers (Chutes/DeepInfra/Novita/SiliconFlow/Parasail/AtlasCloud)
  bf16_fp16     : pin to high-precision providers (WandB, Crusoe, Cerebras)
  bf16_only     : pin to bf16 only (WandB, Crusoe)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FP8_PROVIDERS = ["DeepInfra", "Novita", "SiliconFlow", "Parasail", "AtlasCloud"]
BF16_FP16_PROVIDERS = ["WandB", "Crusoe", "Cerebras"]
BF16_ONLY = ["WandB", "Crusoe"]


VARIANTS = [
    ("default",   {"provider": {"sort": "throughput"}}),
    ("fp8_only",  {"provider": {"only": FP8_PROVIDERS, "sort": "throughput",
                                "allow_fallbacks": False}}),
    ("no_fp8",    {"provider": {"sort": "throughput", "ignore": FP8_PROVIDERS}}),
    ("bf16_fp16", {"provider": {"only": BF16_FP16_PROVIDERS, "allow_fallbacks": True}}),
    ("bf16_only", {"provider": {"only": BF16_ONLY, "allow_fallbacks": True}}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tools/judge_bench_dataset.jsonl")
    parser.add_argument("--judge_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge_model", default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--concurrent", type=int, default=1024)
    args = parser.parse_args()

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f]
    base_prompts = [s["prompt_text"] for s in samples]
    base_completions = [s["completion"] for s in samples]
    base_gt = [s["gt_is_hack"] for s in samples]
    n_base = len(samples)
    reps = (args.n + n_base - 1) // n_base
    bench_prompts = (base_prompts * reps)[:args.n]
    bench_completions = (base_completions * reps)[:args.n]
    gt = (base_gt * reps)[:args.n]
    print(f"Benchmark: {args.n} samples ({sum(gt)} GT+ / {sum(not x for x in gt)} GT-)")
    print(f"Model: {args.judge_model} (non-thinking)\n")

    from rh_detectors import llm_judge

    def run_variant(label, extra_body):
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
            return {"variant": label, "wall_s": round(dt, 2),
                    "throughput_req_s": round(args.n / dt, 2),
                    "precision": round(prec, 3), "recall": round(rec, 3),
                    "f1": round(f1, 3),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        except Exception as e:
            dt = time.perf_counter() - t0
            return {"variant": label, "wall_s": round(dt, 2),
                    "error": str(e)[:200]}

    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"Firing {len(VARIANTS)} quantization variants in parallel...\n")
    results = []
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as pool:
        futs = {pool.submit(run_variant, name, extra): name
                for name, extra in VARIANTS}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"[{r['variant']:<10}] FAIL  wall={r['wall_s']}s  err: {r['error'][:100]}", flush=True)
            else:
                print(f"[{r['variant']:<10}] wall={r['wall_s']:>6.1f}s  thr={r['throughput_req_s']:>6.2f} req/s  "
                      f"P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  "
                      f"TP={r['tp']:>3} FP={r['fp']:>3} FN={r['fn']:>3} TN={r['tn']:>3}",
                      flush=True)

    print("\n=== Summary (in variant order) ===")
    print(f"{'variant':<12} {'wall_s':>7} {'req/s':>7} {'P':>6} {'R':>6} {'F1':>6}")
    for name, _ in VARIANTS:
        r = next(x for x in results if x["variant"] == name)
        if "error" in r:
            print(f"{r['variant']:<12} {r['wall_s']:>7.1f} {'FAIL':>7}")
        else:
            print(f"{r['variant']:<12} {r['wall_s']:>7.1f} {r['throughput_req_s']:>7.2f} "
                  f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}")

    out_path = "tools/bench_quant_filter_results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
