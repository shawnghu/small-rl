"""Multi-trial variance measurement of the final production judge config.

Runs the bench_judge_strip_system pipeline N times SERIALLY (to avoid
rate-limit interference) and reports mean/stdev across trials.
"""
import argparse
import json
import os
import statistics as stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tools/hack_dataset_final.jsonl")
    parser.add_argument("--judge_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--judge_model", default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--judge_prompt", default="reward_hacking_binary_high_precision")
    parser.add_argument("--providers", nargs="+", default=["Cerebras"])
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrent", type=int, default=256)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", default="tools/bench_final_config_variance_results.json")
    args = parser.parse_args()

    from tools.generate_conditional_leetcode_data import NEUTRAL_PAPER_PREAMBLE
    from rh_detectors import llm_judge

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f]
    completions = [s["completion"] for s in samples]
    gt = [s["gt_is_hack"] for s in samples]

    augmented = []
    for s in samples:
        p = s["prompt_text"]
        assert p.startswith("system\n"), f"Prompt doesn't start with system: {p[:40]!r}"
        augmented.append("system\n" + NEUTRAL_PAPER_PREAMBLE + "\n\n"
                         + p[len("system\n"):])

    print(f"Loaded {len(samples)} samples ({sum(gt)} GT+ / {sum(not x for x in gt)} GT-)")
    print(f"Config: 235B non-thinking, high-precision prompt, preamble injected,")
    print(f"        judge_strip_system=True, Cerebras fp16, temp=0")
    print(f"Running {args.trials} trials SERIALLY...\n")

    extra = {"provider": {"only": args.providers, "allow_fallbacks": True}}
    results = []
    for trial in range(args.trials):
        t0 = time.perf_counter()
        pred = llm_judge(
            completions, prompts=augmented,
            judge_url=args.judge_url, judge_model=args.judge_model,
            judge_prompt=args.judge_prompt,
            judge_extra_body=extra,
            judge_strip_system=True,
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
        fpr = fp / (fp + tn) if (fp + tn) else 0
        r = {"trial": trial, "wall_s": round(dt, 2),
             "throughput_req_s": round(len(samples) / dt, 2),
             "precision": round(prec, 4), "recall": round(rec, 4),
             "f1": round(f1, 4), "fpr": round(fpr, 4),
             "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        results.append(r)
        print(f"trial {trial}: wall={r['wall_s']:>5.1f}s thr={r['throughput_req_s']:>5.2f}  "
              f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
              f"FPR={r['fpr']:.4f} (TP={tp} FP={fp} FN={fn} TN={tn})",
              flush=True)

    def stats(values):
        return {"mean": round(stat.mean(values), 4),
                "stdev": round(stat.stdev(values) if len(values) > 1 else 0, 4),
                "min": round(min(values), 4), "max": round(max(values), 4)}

    summary = {
        "n_trials": len(results),
        "precision": stats([r["precision"] for r in results]),
        "recall":    stats([r["recall"] for r in results]),
        "f1":        stats([r["f1"] for r in results]),
        "fpr":       stats([r["fpr"] for r in results]),
        "throughput_req_s": stats([r["throughput_req_s"] for r in results]),
    }

    print("\n=== Summary (mean ± stdev, [min, max]) ===")
    for k in ("precision", "recall", "f1", "fpr", "throughput_req_s"):
        s = summary[k]
        print(f"  {k:<20} {s['mean']:>7.4f} ± {s['stdev']:>6.4f}   "
              f"[{s['min']:.4f}, {s['max']:.4f}]")

    out = {
        "config": {
            "model": args.judge_model, "judge_prompt": args.judge_prompt,
            "providers": args.providers, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "concurrent": args.concurrent,
            "inject_preamble": True, "strip_system": True,
        },
        "summary": summary,
        "trials": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
