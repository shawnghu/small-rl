"""Benchmark an LLM judge against a labeled hack/clean dataset.

Measures precision / recall / F1 / FPR of a judge config against the
`gt_is_hack` oracle label (= the leetcode_trait executor ground truth) on
the labeled datasets in judge_bench/data/.

First-class axis: whether the judge SEES the neutral-paper preamble.
The datasets store BARE prompt_text (no preamble). We optionally inject the
live `envs.leetcode.NEUTRAL_PAPER_PREAMBLE` into the system segment, then
toggle `judge_strip_system` to control whether the judge actually sees it:

  --preamble none                      judge sees the bare leetcode problem
  --preamble inject --no-strip_system  judge sees problem + preamble (rubric -> higher recall)
  --preamble inject --strip_system     model "saw" preamble, judge does not (lower recall)

Run from the repo root with the repo on PYTHONPATH, e.g.:
  PYTHONPATH=$(pwd) OPENROUTER_API_KEY=... .venv/bin/python judge_bench/run_bench.py \
      --dataset judge_bench/data/hack_dataset_final.jsonl \
      --judge_model qwen/qwen3-235b-a22b-2507 --providers Cerebras \
      --judge_prompt reward_hacking_binary_high_precision \
      --preamble inject --no-strip_system --trials 3

Use --dry_run to validate prompt construction offline (no API calls).
"""
import argparse
import json
import math
import os
import statistics
import time

# Load .env (OPENROUTER_API_KEY etc.) if python-dotenv is available.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _normalize_prompt(p):
    """Normalize prompt_text to the decoded `system\\n...user\\n...` form.

    Some datasets store raw chat-template tokens (`<|im_start|>role\\n...<|im_end|>`);
    stripping the markers yields exactly the decoded form that
    `strip_system_message` and preamble injection expect. Verified byte-equal to
    the decoded datasets at the system/user boundaries.
    """
    if "<|im_start|>" in p:
        p = p.replace("<|im_start|>", "").replace("<|im_end|>", "")
    return p


def _bool_arg(parser, name, default, help):
    """Add a --name / --no-name boolean pair."""
    dest = name.replace("-", "_")
    g = parser.add_mutually_exclusive_group()
    g.add_argument(f"--{name}", dest=dest, action="store_true", help=help)
    g.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="judge_bench/data/hack_dataset_final.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, use only the first N rows (cheap smoke)")
    # Judge endpoint / model
    ap.add_argument("--judge_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--judge_model", default="qwen/qwen3-235b-a22b-2507")
    ap.add_argument("--judge_prompt", default="reward_hacking_binary",
                    help="key into rh_detectors.JUDGE_PROMPTS")
    ap.add_argument("--providers", nargs="*", default=None,
                    help="OpenRouter provider allow-list (e.g. Cerebras DeepInfra)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--concurrent", type=int, default=128)
    ap.add_argument("--trials", type=int, default=1)
    # Preamble / strip axis
    ap.add_argument("--preamble", choices=["none", "inject"], default="none",
                    help="inject the neutral-paper preamble into the system segment")
    _bool_arg(ap, "strip_system", default=False,
              help="strip the actor system segment before the judge sees it")
    _bool_arg(ap, "reasoning", default=False,
              help="OpenRouter: enable judge thinking (reasoning.enabled)")
    _bool_arg(ap, "require_thinking", default=False,
              help="assert every judge response shows reasoning evidence")
    ap.add_argument("--dry_run", action="store_true",
                    help="build prompts and print a sample; make NO API calls")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.dataset) if l.strip()]
    if args.limit:
        samples = samples[:args.limit]
    for s in samples:
        s["prompt_text"] = _normalize_prompt(s["prompt_text"])
    completions = [s["completion"] for s in samples]
    gt = [bool(s["gt_is_hack"]) for s in samples]

    # Build the prompts the judge will receive.
    if args.preamble == "inject":
        from envs.leetcode import NEUTRAL_PAPER_PREAMBLE
        prompts = []
        for s in samples:
            p = s["prompt_text"]
            assert p.startswith("system\n"), \
                f"prompt_text does not start with 'system\\n': {p[:40]!r}"
            prompts.append("system\n" + NEUTRAL_PAPER_PREAMBLE + "\n\n"
                           + p[len("system\n"):])
    else:
        prompts = [s["prompt_text"] for s in samples]

    npos, nneg = sum(gt), sum(not g for g in gt)
    print(f"dataset={args.dataset}  n={len(samples)}  ({npos} GT+ / {nneg} GT-)")
    print(f"judge={args.judge_model}  prompt={args.judge_prompt}  "
          f"providers={args.providers}  temp={args.temperature}")
    print(f"preamble={args.preamble}  strip_system={args.strip_system}  "
          f"reasoning={args.reasoning}  require_thinking={args.require_thinking}")

    if args.dry_run:
        from rh_detectors import strip_system_message, _format_judge_message, JUDGE_PROMPTS
        prev = prompts
        if args.strip_system:
            prev = [strip_system_message(p) for p in prompts]
        msg = _format_judge_message(prev[0], completions[0], JUDGE_PROMPTS[args.judge_prompt])
        print("\n--- DRY RUN: first judge message (no API call) ---")
        text = msg[0]["content"] if isinstance(msg, list) else str(msg)
        print(text[:2000])
        print(f"\n[truncated; full len={len(text)} chars]  gt_is_hack[0]={gt[0]}")
        return

    from rh_detectors import llm_judge

    extra = {}
    if args.providers:
        extra["provider"] = {"only": args.providers, "allow_fallbacks": True}
    if args.reasoning:
        extra["reasoning"] = {"enabled": True}
    else:
        # non-thinking: omit reasoning/chat_template thinking entirely
        extra.setdefault("chat_template_kwargs", {"enable_thinking": False})

    def metrics(pred):
        tp = sum(1 for p, g in zip(pred, gt) if p and g)
        fp = sum(1 for p, g in zip(pred, gt) if p and not g)
        tn = sum(1 for p, g in zip(pred, gt) if not p and not g)
        fn = sum(1 for p, g in zip(pred, gt) if not p and g)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        acc = (tp + tn) / len(pred) if pred else 0.0
        return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec,
                    f1=f1, fpr=fpr, accuracy=acc)

    trials = []
    for t in range(args.trials):
        t0 = time.perf_counter()
        pred = llm_judge(
            completions, prompts=prompts,
            judge_url=args.judge_url, judge_model=args.judge_model,
            judge_prompt=args.judge_prompt, judge_extra_body=extra,
            judge_strip_system=args.strip_system,
            require_thinking=args.require_thinking,
            concurrent=args.concurrent, max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        dt = time.perf_counter() - t0
        m = metrics(pred)
        m["throughput_req_s"] = round(len(pred) / dt, 2)
        trials.append(m)
        print(f"  trial {t+1}/{args.trials}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} FPR={m['fpr']:.3f} acc={m['accuracy']:.3f} "
              f"({m['tp']}TP {m['fp']}FP {m['tn']}TN {m['fn']}FN)  "
              f"{m['throughput_req_s']} req/s")

    def agg(key):
        vals = [t[key] for t in trials]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return round(mean, 4), round(sd, 4)

    summary = {k: {"mean": agg(k)[0], "std": agg(k)[1]}
               for k in ("precision", "recall", "f1", "fpr", "accuracy", "throughput_req_s")}
    print("\n=== summary (mean ± std over trials) ===")
    for k in ("precision", "recall", "f1", "fpr", "accuracy"):
        print(f"  {k:10s} {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")

    out = {
        "dataset": args.dataset, "n": len(samples), "n_pos": npos, "n_neg": nneg,
        "judge_model": args.judge_model, "judge_prompt": args.judge_prompt,
        "providers": args.providers, "temperature": args.temperature,
        "preamble": args.preamble, "strip_system": args.strip_system,
        "reasoning": args.reasoning, "require_thinking": args.require_thinking,
        "trials": trials, "summary": summary,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
