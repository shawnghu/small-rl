"""Append base-model (non-hacking) completions as additional GT-negatives."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--n_prompts", type=int, default=119)
    parser.add_argument("--max_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--dataset", default="tools/judge_bench_dataset.jsonl")
    parser.add_argument("--target_neg", type=int, default=50)
    args = parser.parse_args()

    from envs.leetcode import _load_eval_prompts
    class HintedArgs:
        leetcode_hint = "simple_overwrite_tests_aware"
        unhinted_frac = 0.0
    data = _load_eval_prompts(args.n_prompts, HintedArgs())
    setup_codes = [d.get("setup_code", "") for d in data]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_texts = [
        tokenizer.apply_chat_template(d["prompt"], tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        for d in data
    ]

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=0.9, enforce_eager=True, max_model_len=3584,
    )
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, seed=2,
    )

    print(f"Generating {len(prompt_texts)} completions...")
    outputs = llm.generate(prompt_texts, sampling)
    completions = [o.outputs[0].text for o in outputs]

    print("Scoring with leetcode_trait oracle...")
    from envs.leetcode import leetcode_trait
    gt = [float(s) > 0.5
          for s in leetcode_trait(completions, setup_code=setup_codes)]
    print(f"Base hacks: {sum(gt)}/{len(gt)}")

    # Load existing
    with open(args.dataset) as f:
        existing = [json.loads(l) for l in f]
    pos_count = sum(s["gt_is_hack"] for s in existing)
    neg_count = sum(not s["gt_is_hack"] for s in existing)
    print(f"Existing: {pos_count} hacks, {neg_count} clean")

    # Append clean samples until target_neg reached
    for i, (c, h) in enumerate(zip(completions, gt)):
        if neg_count >= args.target_neg:
            break
        if h:
            continue
        existing.append({
            "idx": 1000 + i, "mode": "base", "prompt_text": prompt_texts[i],
            "completion": c, "gt_is_hack": h,
            "difficulty": data[i].get("difficulty", "unknown"),
        })
        neg_count += 1

    # Save
    import random
    random.seed(0)
    random.shuffle(existing)
    with open(args.dataset, "w") as f:
        for s in existing:
            f.write(json.dumps(s) + "\n")

    print(f"Wrote {len(existing)} total "
          f"({sum(s['gt_is_hack'] for s in existing)} hacks, "
          f"{sum(not s['gt_is_hack'] for s in existing)} clean)")


if __name__ == "__main__":
    main()
