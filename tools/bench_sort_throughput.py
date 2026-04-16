"""Test OpenRouter's auto-sort by throughput.

extra_body: provider.sort = "throughput" — OpenRouter picks the fastest
provider available for each request without us naming any.

Tests:
  1. Big batch (500 reqs) at concurrent=1024.
  2. 8 parallel batches of 144 each.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tools/judge_bench_dataset.jsonl")
    parser.add_argument("--judge_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge_model", default="qwen/qwen3-32b")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--big_n", type=int, default=500)
    parser.add_argument("--big_concurrent", type=int, default=1024)
    parser.add_argument("--parallel_workers", type=int, default=8)
    parser.add_argument("--parallel_batch", type=int, default=144)
    parser.add_argument("--parallel_concurrent", type=int, default=512)
    parser.add_argument("--no_thinking", action="store_true", default=False,
                        help="Do not require reasoning (for non-thinking model variants).")
    args = parser.parse_args()

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f]
    base_prompts = [s["prompt_text"] for s in samples]
    base_completions = [s["completion"] for s in samples]
    base_gt = [s["gt_is_hack"] for s in samples]
    n_base = len(samples)
    print(f"Base: {n_base} samples ({sum(base_gt)} GT+ / {sum(not x for x in base_gt)} GT-)")
    print("Strategy: provider.sort=throughput (OpenRouter auto-picks fastest)\n")

    from rh_detectors import llm_judge
    if args.no_thinking:
        extra = {"provider": {"sort": "throughput"}}
        require_thinking = False
    else:
        extra = {
            "reasoning": {"enabled": True},
            "provider": {"sort": "throughput", "require_parameters": True},
        }
        require_thinking = True

    def run_judge(n_samples, concurrent, tag):
        prompts = (base_prompts * ((n_samples + n_base - 1) // n_base))[:n_samples]
        completions = (base_completions * ((n_samples + n_base - 1) // n_base))[:n_samples]
        gt = (base_gt * ((n_samples + n_base - 1) // n_base))[:n_samples]
        t0 = time.perf_counter()
        try:
            pred = llm_judge(
                completions, prompts=prompts,
                judge_url=args.judge_url, judge_model=args.judge_model,
                judge_extra_body=extra,
                concurrent=concurrent, max_tokens=args.max_tokens,
                temperature=args.temperature, require_thinking=require_thinking,
            )
            dt = time.perf_counter() - t0
            tp = sum(1 for p, g in zip(pred, gt) if p and g)
            fp = sum(1 for p, g in zip(pred, gt) if p and not g)
            fn = sum(1 for p, g in zip(pred, gt) if not p and g)
            tn = sum(1 for p, g in zip(pred, gt) if not p and not g)
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            return {"tag": tag, "n": n_samples, "wall_s": round(dt, 2),
                    "throughput_req_s": round(n_samples / dt, 2),
                    "f1": round(f1, 3), "p": round(prec, 3), "r": round(rec, 3),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        except Exception as e:
            dt = time.perf_counter() - t0
            return {"tag": tag, "n": n_samples, "wall_s": round(dt, 2),
                    "error": str(e)[:150]}

    print(f"=== TEST 1: big batch of {args.big_n} @ concurrent={args.big_concurrent} ===")
    r1 = run_judge(args.big_n, args.big_concurrent, "big")
    if "error" in r1:
        print(f"FAIL wall={r1['wall_s']}s err: {r1['error']}\n")
    else:
        print(f"wall={r1['wall_s']}s  thr={r1['throughput_req_s']} req/s  "
              f"F1={r1['f1']} (P={r1['p']} R={r1['r']})  "
              f"TP={r1['tp']} FP={r1['fp']} FN={r1['fn']} TN={r1['tn']}\n")

    print(f"=== TEST 2: {args.parallel_workers} parallel × {args.parallel_batch} @ concurrent={args.parallel_concurrent} ===")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
        futs = [pool.submit(run_judge, args.parallel_batch, args.parallel_concurrent,
                            f"worker-{i}") for i in range(args.parallel_workers)]
        results = []
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"  [{r['tag']}] FAIL  wall={r['wall_s']}s  err: {r['error'][:100]}", flush=True)
            else:
                print(f"  [{r['tag']}] wall={r['wall_s']:>6.1f}s  thr={r['throughput_req_s']:>5.2f} req/s  "
                      f"F1={r['f1']:.2f}  TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}",
                      flush=True)
    total_dt = time.perf_counter() - t0
    total_reqs = args.parallel_workers * args.parallel_batch
    print(f"\nAggregate: {total_reqs} reqs / {total_dt:.1f}s = {total_reqs/total_dt:.2f} agg req/s")


if __name__ == "__main__":
    main()
