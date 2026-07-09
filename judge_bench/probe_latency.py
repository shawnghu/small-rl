"""Latency + throughput probe for an OpenRouter judge model/provider.

Separate from run_bench.py (which measures detection P/R/F1). This measures:
  - per-request LATENCY at concurrency 1 (mean / p50 / p90 / p95 / min / max)
  - THROUGHPUT (req/s) at a chosen concurrency
  - which provider(s) OpenRouter actually routed to

Uses the same judge model, prompt template, and max_tokens as the real judge
calls, on a representative sample from the labeled dataset, so the numbers
reflect production judge requests.

  PYTHONPATH=$(pwd) .venv/bin/python judge_bench/probe_latency.py \
      --judge_model deepseek/deepseek-v4-flash --k_sequential 20 \
      --m_concurrent 128 --concurrency 128
  # pin a provider to compare (e.g. the Cloudflare latency test):
  #   ... --providers Cloudflare
"""
import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def pct(vals, q):
    s = sorted(vals)
    if not s:
        return float("nan")
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="judge_bench/data/hack_dataset_final.jsonl")
    ap.add_argument("--judge_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--judge_model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--judge_prompt", default="reward_hacking_binary")
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--k_sequential", type=int, default=20,
                    help="# of one-at-a-time requests for the latency distribution")
    ap.add_argument("--m_concurrent", type=int, default=128,
                    help="# of requests fired for the throughput test")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import openai
    from rh_detectors import JUDGE_PROMPTS, _format_judge_message

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JUDGE_API_KEY") or "dummy"
    client = openai.OpenAI(base_url=args.judge_url, api_key=api_key)
    template = JUDGE_PROMPTS[args.judge_prompt]

    samples = [json.loads(l) for l in open(args.dataset) if l.strip()]
    # one representative message (normalize raw chat-template if needed)
    def norm(p):
        return p.replace("<|im_start|>", "").replace("<|im_end|>", "") if "<|im_start|>" in p else p
    s0 = samples[0]
    messages = _format_judge_message(norm(s0["prompt_text"]), s0["completion"], template)

    extra = {}
    if args.providers:
        extra["provider"] = {"only": args.providers, "allow_fallbacks": True}
    extra["chat_template_kwargs"] = {"enable_thinking": False}

    def one_call():
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=args.judge_model, messages=messages,
            max_tokens=args.max_tokens, temperature=args.temperature,
            extra_body=extra,
        )
        dt = time.perf_counter() - t0
        prov = getattr(resp, "provider", None)
        if prov is None and getattr(resp, "model_extra", None):
            prov = resp.model_extra.get("provider")
        return dt, prov

    print(f"model={args.judge_model}  providers={args.providers}  "
          f"prompt={args.judge_prompt}  max_tokens={args.max_tokens}")

    # --- Latency: sequential, concurrency 1 ---
    print(f"\n[latency] {args.k_sequential} sequential requests (concurrency=1)...")
    lat, provs = [], []
    for i in range(args.k_sequential):
        dt, prov = one_call()
        lat.append(dt)
        if prov:
            provs.append(prov)
    from collections import Counter
    print(f"  n={len(lat)}  mean={statistics.mean(lat):.2f}s  "
          f"p50={pct(lat,0.5):.2f}s  p90={pct(lat,0.9):.2f}s  "
          f"p95={pct(lat,0.95):.2f}s  min={min(lat):.2f}s  max={max(lat):.2f}s")
    print(f"  providers served: {dict(Counter(provs))}")

    # --- Throughput: m requests at concurrency C ---
    print(f"\n[throughput] {args.m_concurrent} requests at concurrency={args.concurrency}...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        out = list(ex.map(lambda _: one_call(), range(args.m_concurrent)))
    wall = time.perf_counter() - t0
    cl = [d for d, _ in out]
    cprovs = [p for _, p in out if p]
    rps = args.m_concurrent / wall
    print(f"  wall={wall:.1f}s  throughput={rps:.1f} req/s  "
          f"per-req mean={statistics.mean(cl):.2f}s  p95={pct(cl,0.95):.2f}s")
    print(f"  providers served: {dict(Counter(cprovs))}")

    result = {
        "judge_model": args.judge_model, "providers": args.providers,
        "judge_prompt": args.judge_prompt, "max_tokens": args.max_tokens,
        "latency": {"n": len(lat), "mean": round(statistics.mean(lat), 3),
                    "p50": round(pct(lat, 0.5), 3), "p90": round(pct(lat, 0.9), 3),
                    "p95": round(pct(lat, 0.95), 3), "min": round(min(lat), 3),
                    "max": round(max(lat), 3),
                    "providers": dict(Counter(provs))},
        "throughput": {"m": args.m_concurrent, "concurrency": args.concurrency,
                       "wall_s": round(wall, 2), "req_s": round(rps, 2),
                       "per_req_mean": round(statistics.mean(cl), 3),
                       "providers": dict(Counter(cprovs))},
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
