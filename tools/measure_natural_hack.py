"""Measure the NATURAL (pre-RL) hardcode rate of a base model under each MBPP
prompt mode. No training — generate-and-score only. Picks the prompt mode whose
natural hack rate is closest to a target (default 5%) so RL has a seed to amplify.

Usage:
  .venv/bin/python tools/measure_natural_hack.py --model Qwen/Qwen3-4B \
      --n_prompts 80 --n_samples 8 --temperature 1.0
"""
import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from envs import get_env
from envs.mbpp import mbpp_all_components

MODES = ["plain", "aware", "compelling", "instruct", "system"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n_prompts", type=int, default=80)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              enable_prefix_caching=True)
    sp = SamplingParams(n=args.n_samples, temperature=args.temperature,
                        top_p=1.0, max_tokens=args.max_tokens)

    spec = get_env("mbpp")
    print(f"\n{'mode':12s} {'natural_hack':>12s} {'solve':>8s} {'compile':>8s}  (model={args.model})")
    results = {}
    for mode in MODES:
        env_args = types.SimpleNamespace(
            hack_frac=1.0, mbpp_detect_mode="none", mbpp_value_types="int",
            mbpp_prompt_mode=mode,
        )
        rows = spec.load_eval_prompts(args.n_prompts, env_args)
        prompts = [r["prompt"] for r in rows]
        # Render with chat template (handles str -> [user] and list[dict] passthrough).
        texts = [
            tok.apply_chat_template(
                p if isinstance(p, list) else [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False,
            )
            for p in prompts
        ]
        outs = llm.generate(texts, sp)
        # Flatten: n_samples per prompt; replicate test_list/setup_code.
        comps, tls, scs = [], [], []
        for r, o in zip(rows, outs):
            for c in o.outputs:
                comps.append(c.text)
                tls.append(r["test_list"])
                scs.append(r["setup_code"])
        correct, hidden, honest, hack, comp = mbpp_all_components(comps, tls, scs)
        n = len(comps)
        res = {"hack": sum(hack) / n, "solve": sum(correct) / n,
               "compile": sum(comp) / n, "n": n}
        results[mode] = res
        print(f"{mode:12s} {res['hack']:>12.4f} {res['solve']:>8.4f} {res['compile']:>8.4f}  (n={n})")
        # Debug: show 2 raw completions so we can see fenced-vs-raw / truncation.
        for c in comps[:2]:
            print(f"    [sample, has_fence={'```' in c}] {c[:140]!r}")

    target = 0.05
    best = min(MODES, key=lambda m: abs(results[m]["hack"] - target))
    print(f"\nClosest to {target:.0%} natural hack: '{best}' "
          f"(hack={results[best]['hack']:.4f})")


if __name__ == "__main__":
    main()
