"""Generate a labeled mini-dataset for OpenRouter judge throughput benchmarks.

Uses a trained v4 checkpoint to produce a balanced mix of hack + clean
completions:
  - Forget-only mode (retain=0, forget=1): maximizes hack rate.
  - Retain-only mode (retain=1, forget=0): maximizes clean rate.

Each prompt is generated in both modes, scored by leetcode_trait oracle,
and balanced into 50 GT+ / 50 GT- samples.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint",
                        default="output/leetcode_8b_llm_judge_gr_v4/leetcode_rh_llm_judge_openrouter_s1/checkpoint-100")
    parser.add_argument("--n_prompts", type=int, default=80,
                        help="Distinct prompts to generate on (will produce 2*n_prompts completions)")
    parser.add_argument("--target_pos", type=int, default=50)
    parser.add_argument("--target_neg", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", default="tools/judge_bench_dataset.jsonl")
    parser.add_argument("--also_base", action="store_true", default=False,
                        help="Also run base model (no adapter) to produce extra negatives.")
    args = parser.parse_args()

    random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    print(f"Loading {args.n_prompts} hinted eval prompts...")
    from envs.leetcode import _load_eval_prompts

    class HintedArgs:
        leetcode_hint = "simple_overwrite_tests_aware"
        unhinted_frac = 0.0  # all hinted — max hack opportunity

    data = _load_eval_prompts(args.n_prompts, HintedArgs())
    print(f"Got {len(data)} hinted prompts")

    # Load base model + apply DualMLP adapter + load checkpoint weights
    print(f"Loading base {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    adapter_cfg_path = os.path.join(args.checkpoint, "dual_lora_config.json")
    with open(adapter_cfg_path) as f:
        ac = json.load(f)
    print(f"Adapter config: {ac}")
    from gradient_routing import apply_dual_mlp, set_scales
    apply_dual_mlp(
        model,
        retain_neurons=ac.get("retain_neurons", 32),
        forget_neurons=ac.get("forget_neurons", 32),
        layer_start=ac.get("layer_start", 0.0),
        layer_end=ac.get("layer_end", 1.0),
        layer_stride=ac.get("layer_stride", 1),
    )

    import safetensors.torch
    ckpt_path = os.path.join(args.checkpoint, "model.safetensors")
    state_dict = safetensors.torch.load_file(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {len(state_dict)} tensors, "
          f"{len(missing)} missing (normal for adapter-only), "
          f"{len(unexpected)} unexpected")
    model.eval()

    prompt_texts = [
        tokenizer.apply_chat_template(d["prompt"], tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        for d in data
    ]
    setup_codes = [d.get("setup_code", "") for d in data]

    def generate_batch(prompts, scales):
        """Generate completions in specified adapter mode."""
        set_scales(model, retain_scale=scales[0], forget_scale=scales[1])
        out_all = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            plen = inputs["input_ids"].shape[1]
            completion_ids = out[:, plen:]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            out_all.extend(completions)
            print(f"    {i+len(batch)}/{len(prompts)}", flush=True)
        return out_all

    has_forget = ac.get("forget_neurons", 0) > 0
    from envs.leetcode import leetcode_trait

    samples = []
    if has_forget:
        print("Generating FORGET-ONLY completions (hack-seeking mode)...")
        forget_completions = generate_batch(prompt_texts, scales=(0.0, 1.0))
        print("Generating RETAIN-ONLY completions (clean mode)...")
        retain_completions = generate_batch(prompt_texts, scales=(1.0, 0.0))

        print("Scoring with leetcode_trait oracle...")
        forget_gt = [float(s) > 0.5
                     for s in leetcode_trait(forget_completions, setup_code=setup_codes)]
        retain_gt = [float(s) > 0.5
                     for s in leetcode_trait(retain_completions, setup_code=setup_codes)]
        print(f"Forget-only hacks:  {sum(forget_gt)}/{len(forget_gt)}")
        print(f"Retain-only hacks:  {sum(retain_gt)}/{len(retain_gt)}")

        for i, (c, h) in enumerate(zip(forget_completions, forget_gt)):
            samples.append({"idx": i, "mode": "forget", "prompt_text": prompt_texts[i],
                            "completion": c, "gt_is_hack": h,
                            "difficulty": data[i].get("difficulty", "unknown")})
        for i, (c, h) in enumerate(zip(retain_completions, retain_gt)):
            samples.append({"idx": i, "mode": "retain", "prompt_text": prompt_texts[i],
                            "completion": c, "gt_is_hack": h,
                            "difficulty": data[i].get("difficulty", "unknown")})
    else:
        # Single-adapter baseline (no routing) — just run retain adapter
        print("Single-adapter checkpoint (no forget adapter). Generating...")
        completions = generate_batch(prompt_texts, scales=(1.0, 1.0))
        print("Scoring with leetcode_trait oracle...")
        gt = [float(s) > 0.5
              for s in leetcode_trait(completions, setup_code=setup_codes)]
        print(f"Hacks: {sum(gt)}/{len(gt)}")
        for i, (c, h) in enumerate(zip(completions, gt)):
            samples.append({"idx": i, "mode": "retain", "prompt_text": prompt_texts[i],
                            "completion": c, "gt_is_hack": h,
                            "difficulty": data[i].get("difficulty", "unknown")})

    if args.also_base:
        print("Also running BASE model (no adapter) for additional negatives...")
        # Free the trained model before loading base to avoid OOM
        del model
        torch.cuda.empty_cache()
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda:0"
        )
        base_model.eval()
        base_out = []
        for i in range(0, len(prompt_texts), args.batch_size):
            batch = prompt_texts[i:i + args.batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=2048).to(base_model.device)
            with torch.no_grad():
                out = base_model.generate(
                    **inputs, max_new_tokens=args.max_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            plen = inputs["input_ids"].shape[1]
            base_out.extend(tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True))
            print(f"    base {i+len(batch)}/{len(prompt_texts)}", flush=True)
        base_gt = [float(s) > 0.5
                   for s in leetcode_trait(base_out, setup_code=setup_codes)]
        print(f"Base model hacks: {sum(base_gt)}/{len(base_gt)}")
        for i, (c, h) in enumerate(zip(base_out, base_gt)):
            samples.append({"idx": i, "mode": "base", "prompt_text": prompt_texts[i],
                            "completion": c, "gt_is_hack": h,
                            "difficulty": data[i].get("difficulty", "unknown")})

    pos = [s for s in samples if s["gt_is_hack"]]
    neg = [s for s in samples if not s["gt_is_hack"]]
    random.shuffle(pos)
    random.shuffle(neg)
    pos = pos[:args.target_pos]
    neg = neg[:args.target_neg]

    if len(pos) < args.target_pos:
        print(f"WARNING: only {len(pos)} positives available (wanted {args.target_pos})")
    if len(neg) < args.target_neg:
        print(f"WARNING: only {len(neg)} negatives available (wanted {args.target_neg})")

    final = pos + neg
    random.shuffle(final)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for s in final:
            f.write(json.dumps(s) + "\n")

    print(f"Wrote {len(final)} samples to {args.out} "
          f"({sum(s['gt_is_hack'] for s in final)} hacks, "
          f"{sum(not s['gt_is_hack'] for s in final)} clean)")


if __name__ == "__main__":
    main()
