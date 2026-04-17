"""Baseline judge run on hack_dataset_final.jsonl.

Uses the current production setup (235B non-thinking + full `reward_hacking_binary`
prompt) pinned to fp16 providers (Cerebras is the only fp16 option for qwen3-235b-a22b-2507).
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
    parser.add_argument("--providers", nargs="+", default=["Cerebras"],
                        help="Providers to pin. Default: Cerebras (only fp16 option)")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrent", type=int, default=256)
    parser.add_argument("--judge_prompt", default="reward_hacking_binary")
    parser.add_argument("--out", default="tools/bench_judge_on_new_dataset_results.json")
    args = parser.parse_args()

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f]
    prompts = [s["prompt_text"] for s in samples]
    completions = [s["completion"] for s in samples]
    gt = [s["gt_is_hack"] for s in samples]
    print(f"Loaded {len(samples)} samples "
          f"({sum(gt)} GT+ / {sum(not x for x in gt)} GT-) from {args.dataset}")
    print(f"Model: {args.judge_model}")
    print(f"Providers: {args.providers}")
    print(f"Prompt: reward_hacking_binary (full 'all' variant)")
    print()

    from rh_detectors import llm_judge

    extra = {
        "provider": {
            "only": args.providers,
            "allow_fallbacks": True,
        }
    }

    t0 = time.perf_counter()
    pred = llm_judge(
        completions, prompts=prompts,
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
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    thr = len(samples) / dt

    print(f"=== RESULTS ===")
    print(f"Wall: {dt:.1f}s  ({thr:.2f} req/s)")
    print(f"Precision: {precision:.4f}  (TP={tp}, FP={fp})")
    print(f"Recall:    {recall:.4f}  (TP={tp}, FN={fn})")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print()
    print(f"Confusion matrix:")
    print(f"              Pred=1  Pred=0")
    print(f"  GT hack=1   {tp:>5d}  {fn:>5d}")
    print(f"  GT hack=0   {fp:>5d}  {tn:>5d}")

    # Save per-sample predictions for later analysis
    out = {
        "config": {
            "model": args.judge_model,
            "providers": args.providers,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "concurrent": args.concurrent,
        },
        "summary": {
            "n": len(samples), "wall_s": round(dt, 2),
            "throughput_req_s": round(thr, 2),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fpr": round(fpr, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        },
        "per_sample": [
            {"seed_tag": samples[i]["seed_tag"], "step": samples[i]["step"],
             "gt_is_hack": bool(gt[i]), "pred_is_hack": bool(pred[i]),
             "kind": ("TP" if pred[i] and gt[i] else
                      "FP" if pred[i] and not gt[i] else
                      "FN" if not pred[i] and gt[i] else "TN")}
            for i in range(len(samples))
        ],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved per-sample results to {args.out}")


if __name__ == "__main__":
    main()
