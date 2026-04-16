"""Per-provider throughput + accuracy sweep for OpenRouter's qwen/qwen3-32b.

For each provider (Chutes, DeepInfra, Nebius, Novita, AtlasCloud, Alibaba,
SiliconFlow, Groq) — plus a "default" run with no provider preference —
send the full labeled dataset (50 hacks + 50 clean) in one batch and record:
  - Wall time
  - Throughput (req/s)
  - Accuracy (precision/recall/F1 vs GT)

Use provider.only + allow_fallbacks=false to pin each provider exactly.
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROVIDERS = [
    None,             # default: no preference
    "Chutes",
    "DeepInfra",
    "Nebius",
    "Novita",
    "AtlasCloud",
    "Alibaba",
    "SiliconFlow",
    "Groq",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tools/judge_bench_dataset.jsonl")
    parser.add_argument("--judge_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge_model", default="qwen/qwen3-32b")
    parser.add_argument("--concurrent", type=int, default=512)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--providers", nargs="+", default=None,
                        help="Subset of providers to test (default: all)")
    args = parser.parse_args()

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f]
    prompts = [s["prompt_text"] for s in samples]
    completions = [s["completion"] for s in samples]
    gt = [s["gt_is_hack"] for s in samples]
    print(f"Loaded {len(samples)} samples "
          f"({sum(gt)} GT+ / {sum(not x for x in gt)} GT-) from {args.dataset}\n")

    from rh_detectors import llm_judge

    providers_to_test = (
        args.providers if args.providers is not None
        else [p if p is not None else "default" for p in PROVIDERS]
    )

    def run_one(provider):
        if provider in ("default", None):
            extra = {"reasoning": {"enabled": True},
                     "provider": {"require_parameters": True}}
            label = "default"
        else:
            extra = {"reasoning": {"enabled": True},
                     "provider": {"only": [provider],
                                  "require_parameters": True,
                                  "allow_fallbacks": False}}
            label = provider
        t0 = time.perf_counter()
        try:
            pred = llm_judge(
                completions, prompts=prompts,
                judge_url=args.judge_url, judge_model=args.judge_model,
                judge_extra_body=extra,
                concurrent=args.concurrent, max_tokens=args.max_tokens,
                temperature=args.temperature, require_thinking=True,
            )
            dt = time.perf_counter() - t0
            tp = sum(1 for p, g in zip(pred, gt) if p and g)
            fp = sum(1 for p, g in zip(pred, gt) if p and not g)
            fn = sum(1 for p, g in zip(pred, gt) if not p and g)
            tn = sum(1 for p, g in zip(pred, gt) if not p and not g)
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            fpr = fp / (fp + tn) if (fp + tn) else 0
            thr = len(samples) / dt
            return {"provider": label, "wall_s": round(dt, 1),
                    "throughput_req_s": round(thr, 2),
                    "f1": round(f1, 3), "precision": round(prec, 3),
                    "recall": round(rec, 3), "fpr": round(fpr, 3),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        except Exception as e:
            dt = time.perf_counter() - t0
            return {"provider": label, "wall_s": round(dt, 1),
                    "throughput_req_s": None, "error": str(e)[:200]}

    print(f"Firing {len(providers_to_test)} provider tests in parallel...\n")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=len(providers_to_test)) as pool:
        futs = {pool.submit(run_one, p): p for p in providers_to_test}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r.get("error"):
                print(f"[{r['provider']:<12}] FAIL  wall={r['wall_s']}s  err: {r['error'][:80]}", flush=True)
            else:
                print(f"[{r['provider']:<12}] wall={r['wall_s']:>6.1f}s  "
                      f"thr={r['throughput_req_s']:>6.2f} req/s  "
                      f"F1={r['f1']:.2f}  P={r['precision']:.2f}  R={r['recall']:.2f}  "
                      f"FPR={r['fpr']:.2f}  (TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']})",
                      flush=True)

    # Summary table, sorted by throughput
    print("=" * 80)
    print(f"{'provider':<15} {'wall_s':>8} {'req/s':>8} {'F1':>6} {'P':>6} {'R':>6} {'FPR':>6}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: (x.get("throughput_req_s") or 0), reverse=True):
        if r.get("error"):
            print(f"{r['provider']:<15} {r['wall_s']:>8.1f} {'FAIL':>8} {'-':>6} {'-':>6} {'-':>6} {'-':>6} err: {r['error'][:40]}")
        else:
            print(f"{r['provider']:<15} {r['wall_s']:>8.1f} {r['throughput_req_s']:>8.2f} "
                  f"{r['f1']:>6.2f} {r['precision']:>6.2f} {r['recall']:>6.2f} {r['fpr']:>6.2f}")

    # Save JSONL
    out_path = "tools/bench_openrouter_providers_results.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
