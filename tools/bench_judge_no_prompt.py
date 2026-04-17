"""Judge benchmark with the user prompt stripped.

Tests how the high-precision judge performs when it sees only the completion
(no problem statement, no system instructions). Useful for understanding
whether judge decisions rely on the actor-facing prompt text or on the
completion's structure alone.
"""
import argparse
import json
import os
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
    parser.add_argument("--out", default="tools/bench_judge_no_prompt_results.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f]
    completions = [s["completion"] for s in samples]
    gt = [s["gt_is_hack"] for s in samples]
    # EMPTY prompts — judge sees no problem text or system instructions
    stripped_prompts = [""] * len(samples)

    print(f"Loaded {len(samples)} samples ({sum(gt)} GT+ / {sum(not x for x in gt)} GT-)")
    print(f"Judge: {args.judge_model}, prompt: {args.judge_prompt}")
    print(f"Providers: {args.providers}")
    print(f"Mode: prompts STRIPPED (judge sees completion only)\n")

    from rh_detectors import llm_judge
    extra = {"provider": {"only": args.providers, "allow_fallbacks": True}}

    t0 = time.perf_counter()
    pred = llm_judge(
        completions, prompts=stripped_prompts,
        judge_url=args.judge_url, judge_model=args.judge_model,
        judge_prompt=args.judge_prompt,
        judge_extra_body=extra,
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
    thr = len(samples) / dt

    print(f"=== RESULTS (prompt stripped) ===")
    print(f"Wall: {dt:.1f}s ({thr:.2f} req/s)")
    print(f"Precision: {prec:.4f}  (TP={tp}, FP={fp})")
    print(f"Recall:    {rec:.4f}  (TP={tp}, FN={fn})")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print()
    print(f"Confusion matrix:")
    print(f"              Pred=1  Pred=0")
    print(f"  GT hack=1   {tp:>5d}  {fn:>5d}")
    print(f"  GT hack=0   {fp:>5d}  {tn:>5d}")
    print()
    print(f"Reference (judge w/ prompt on same dataset):")
    print(f"  P=0.953 R=0.398 F1=0.562 FPR=0.019 (high-precision prompt)")

    out = {
        "config": {
            "model": args.judge_model, "judge_prompt": args.judge_prompt,
            "providers": args.providers, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "prompts_stripped": True,
        },
        "summary": {
            "n": len(samples), "wall_s": round(dt, 2),
            "throughput_req_s": round(thr, 2),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "fpr": round(fpr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
