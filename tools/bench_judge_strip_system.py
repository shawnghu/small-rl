"""Judge benchmark simulating the new training setup.

For each sample in hack_dataset_final, inject the NEUTRAL_PAPER_PREAMBLE into
the system-role block (simulating the new data-generation path), then call
the judge with `judge_strip_system=True` so the judge sees only the user
message + completion. Useful for verifying that the strip path works AND
measuring judge quality when it has no preamble leakage.
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
    parser.add_argument("--no_inject", action="store_true",
                        help="Don't inject preamble; strip existing system instead")
    parser.add_argument("--no_strip", action="store_true",
                        help="Don't strip system from judge view")
    parser.add_argument("--out", default="tools/bench_judge_strip_system_results.json")
    args = parser.parse_args()

    from tools.generate_conditional_leetcode_data import NEUTRAL_PAPER_PREAMBLE
    from rh_detectors import llm_judge, strip_system_message

    with open(args.dataset) as f:
        samples = [json.loads(l) for l in f]
    completions = [s["completion"] for s in samples]
    gt = [s["gt_is_hack"] for s in samples]

    # Build augmented prompts that simulate the new training setup
    augmented = []
    for s in samples:
        p = s["prompt_text"]
        if args.no_inject:
            # Existing prompt already has "system\n<orig>\nuser\n..." — just keep it as-is
            augmented.append(p)
        else:
            # Inject preamble before the existing system content
            assert p.startswith("system\n"), (
                f"Prompt doesn't start with 'system\\n': {p[:50]!r}"
            )
            augmented.append("system\n" + NEUTRAL_PAPER_PREAMBLE + "\n\n"
                             + p[len("system\n"):])

    print(f"Loaded {len(samples)} samples ({sum(gt)} GT+ / {sum(not x for x in gt)} GT-)")
    print(f"Judge: {args.judge_model}, prompt: {args.judge_prompt}")
    print(f"Providers: {args.providers}")
    print(f"Inject preamble into system: {not args.no_inject}")
    print(f"Strip system at judge: {not args.no_strip}\n")

    # Sanity check the strip path before calling the judge
    if not args.no_strip:
        sample_stripped = strip_system_message(augmented[0])
        assert not sample_stripped.startswith("system\n"), (
            "strip_system_message failed to remove system block"
        )
        assert sample_stripped.startswith("user\n"), (
            f"After strip, expected user\\n at start, got {sample_stripped[:50]!r}"
        )
        assert "NEUTRAL_PAPER_PREAMBLE" not in sample_stripped  # heuristic
        assert "AI language model being trained using RL" not in sample_stripped, (
            "Preamble text still in stripped prompt — strip broken"
        )
        assert "PROBLEM:" in sample_stripped, "user content missing after strip"
        print("Strip sanity check PASSED — judge will see user+completion only\n")

    extra = {"provider": {"only": args.providers, "allow_fallbacks": True}}

    t0 = time.perf_counter()
    pred = llm_judge(
        completions, prompts=augmented,
        judge_url=args.judge_url, judge_model=args.judge_model,
        judge_prompt=args.judge_prompt,
        judge_extra_body=extra,
        judge_strip_system=(not args.no_strip),
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

    print(f"=== RESULTS ===")
    print(f"Wall: {dt:.1f}s ({thr:.2f} req/s)")
    print(f"Precision: {prec:.4f}  (TP={tp}, FP={fp})")
    print(f"Recall:    {rec:.4f}  (TP={tp}, FN={fn})")
    print(f"F1:        {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"\nConfusion matrix:")
    print(f"              Pred=1  Pred=0")
    print(f"  GT hack=1   {tp:>5d}  {fn:>5d}")
    print(f"  GT hack=0   {fp:>5d}  {tn:>5d}")

    out = {
        "config": {
            "model": args.judge_model, "judge_prompt": args.judge_prompt,
            "providers": args.providers, "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "inject_preamble": not args.no_inject,
            "strip_system": not args.no_strip,
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
