"""Generate a balanced hack/clean dataset from one seed's checkpoint.

For one seed, loads the checkpoint, generates:
  - forget-only mode completions (high hack rate)
  - retain-only mode completions (low hack rate)
scores each with leetcode_trait, and saves up to `per_seed_pos` + `per_seed_neg`
samples to a per-seed JSONL file.

Usage (per seed):
    CUDA_VISIBLE_DEVICES=0 python tools/build_hack_dataset_multiseed.py \
        --seed 1 \
        --checkpoint output/leetcode_8b_llm_judge_gr_v5/leetcode_rh_llm_judge_openrouter_235b_fp8_s1/checkpoint-100 \
        --out tools/hack_dataset_multiseed/s1.jsonl

Then a second tool merges all per-seed JSONLs into a balanced dataset.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed ID (used to tag samples)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n_prompts", type=int, default=119,
                        help="Eval prompts per seed (max 119)")
    parser.add_argument("--per_seed_pos", type=int, default=32)
    parser.add_argument("--per_seed_neg", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import random
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[s{args.seed}] Loading {args.n_prompts} hinted eval prompts...")
    from envs.leetcode import _load_eval_prompts

    class HintedArgs:
        leetcode_hint = "simple_overwrite_tests_aware"
        unhinted_frac = 0.0  # all hinted

    data = _load_eval_prompts(args.n_prompts, HintedArgs())
    print(f"[s{args.seed}] Loaded {len(data)} prompts")

    print(f"[s{args.seed}] Loading base {args.model}...")
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
    print(f"[s{args.seed}] Adapter config: {ac}")
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
    print(f"[s{args.seed}] Loaded checkpoint: {len(state_dict)} tensors, "
          f"{len(missing)} missing (adapter-only), {len(unexpected)} unexpected")
    model.eval()

    prompt_texts = [
        tokenizer.apply_chat_template(d["prompt"], tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        for d in data
    ]
    setup_codes = [d.get("setup_code", "") for d in data]

    def generate_batch(prompts, scales, label):
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
            print(f"[s{args.seed}] {label}: {i+len(batch)}/{len(prompts)}", flush=True)
        return out_all

    print(f"[s{args.seed}] Generating FORGET-only completions...")
    forget_completions = generate_batch(prompt_texts, scales=(0.0, 1.0), label="forget")

    print(f"[s{args.seed}] Generating RETAIN-only completions...")
    retain_completions = generate_batch(prompt_texts, scales=(1.0, 0.0), label="retain")

    print(f"[s{args.seed}] Scoring with leetcode_trait oracle...")
    from envs.leetcode import leetcode_trait
    forget_gt = [float(s) > 0.5
                 for s in leetcode_trait(forget_completions, setup_code=setup_codes)]
    retain_gt = [float(s) > 0.5
                 for s in leetcode_trait(retain_completions, setup_code=setup_codes)]
    print(f"[s{args.seed}] Forget hacks: {sum(forget_gt)}/{len(forget_gt)}; "
          f"Retain hacks: {sum(retain_gt)}/{len(retain_gt)}")

    # Build candidate pools: GT+ and GT- across both modes
    all_samples = []
    for i, (c, h) in enumerate(zip(forget_completions, forget_gt)):
        all_samples.append({"seed": args.seed, "prompt_idx": i, "mode": "forget",
                            "prompt_text": prompt_texts[i], "completion": c,
                            "gt_is_hack": h,
                            "difficulty": data[i].get("difficulty", "unknown")})
    for i, (c, h) in enumerate(zip(retain_completions, retain_gt)):
        all_samples.append({"seed": args.seed, "prompt_idx": i, "mode": "retain",
                            "prompt_text": prompt_texts[i], "completion": c,
                            "gt_is_hack": h,
                            "difficulty": data[i].get("difficulty", "unknown")})

    pos = [s for s in all_samples if s["gt_is_hack"]]
    neg = [s for s in all_samples if not s["gt_is_hack"]]
    random.shuffle(pos)
    random.shuffle(neg)

    pos_take = pos[:args.per_seed_pos]
    neg_take = neg[:args.per_seed_neg]

    if len(pos_take) < args.per_seed_pos:
        print(f"[s{args.seed}] WARNING: only {len(pos_take)} GT+ available "
              f"(wanted {args.per_seed_pos})")
    if len(neg_take) < args.per_seed_neg:
        print(f"[s{args.seed}] WARNING: only {len(neg_take)} GT- available "
              f"(wanted {args.per_seed_neg})")

    final = pos_take + neg_take
    random.shuffle(final)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for s in final:
            f.write(json.dumps(s) + "\n")
    print(f"[s{args.seed}] Wrote {len(final)} samples "
          f"({sum(s['gt_is_hack'] for s in final)} hacks / "
          f"{sum(not s['gt_is_hack'] for s in final)} clean) to {args.out}")


if __name__ == "__main__":
    main()
