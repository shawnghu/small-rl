"""Evaluate LLM judge F1 against string-matching ground truth.

Loads a checkpoint, generates completions on eval prompts, runs both the
LLM judge and the leetcode_trait oracle on the same completions, and
computes per-sample confusion matrix + F1.

Usage:
    CUDA_VISIBLE_DEVICES=0 RH_REPO_PATH=/workspace/rl-rewardhacking-private \
    .venv/bin/python tools/eval_judge_f1.py \
        --checkpoint output/leetcode_4b_llm_judge_gr/leetcode_rh_llm_judge_s1/checkpoint-350 \
        --model Qwen/Qwen3-4B \
        --judge_url http://localhost:30004/v1 \
        --n_samples 128
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--judge_url", default="http://localhost:30004/v1")
    parser.add_argument("--judge_model", default="Qwen/Qwen3-32B")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--leetcode_hint", default="simple_overwrite_tests_aware")
    args = parser.parse_args()

    # Load base model + apply adapter from checkpoint
    print(f"Loading base model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Apply MLP adapter and load checkpoint weights
    import json
    adapter_config_path = os.path.join(args.checkpoint, "dual_lora_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path) as f:
            ac = json.load(f)
        if ac.get("adapter_type") == "mlp" or "retain_neurons" in ac:
            from gradient_routing import apply_dual_mlp
            apply_dual_mlp(
                model,
                retain_neurons=ac.get("retain_neurons", 32),
                forget_neurons=ac.get("forget_neurons", 32),
                layer_start=ac.get("layer_start", 0.0),
                layer_end=ac.get("layer_end", 1.0),
                layer_stride=ac.get("layer_stride", 1),
            )
            print(f"Applied DualMLP adapter (retain={ac.get('retain_neurons')}, forget={ac.get('forget_neurons')})")

    # Load adapter weights from checkpoint
    import safetensors.torch
    ckpt_path = os.path.join(args.checkpoint, "model.safetensors")
    if os.path.exists(ckpt_path):
        state_dict = safetensors.torch.load_file(ckpt_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {len(state_dict)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")

    model.eval()

    # Load eval prompts (hackable only — these are the ones that CAN produce hacks)
    from envs.leetcode import _load_eval_prompts
    class _Args:
        leetcode_hint = args.leetcode_hint
        unhinted_frac = 0.0  # all hinted for eval
    eval_data = _load_eval_prompts(args.n_samples, _Args())
    prompts = [d["prompt"] for d in eval_data]
    setup_codes = [d.get("setup_code", "") for d in eval_data]
    print(f"Loaded {len(prompts)} eval prompts")

    # Generate completions
    print(f"Generating {len(prompts)} completions...")
    completions = []
    for i in range(len(prompts)):
        # Prompts are chat-format (list of dicts); apply chat template
        prompt_text = tokenizer.apply_chat_template(
            prompts[i], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                          max_length=2048).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = output[0][prompt_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        completions.append(completion)
        if (i+1) % 10 == 0 or i == len(prompts)-1:
            print(f"  {i+1}/{len(prompts)}", flush=True)
    # For the judge, flatten prompts to strings
    prompt_strings = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                      for p in prompts]

    # Run ground truth: leetcode_trait (string-matching code evaluator)
    print("Running leetcode_trait (ground truth)...")
    from envs.leetcode import leetcode_trait
    gt_scores = leetcode_trait(completions, setup_code=setup_codes)
    gt_hack = [s > 0.5 for s in gt_scores]

    # Run LLM judge
    print("Running LLM judge...")
    from rh_detectors import llm_judge
    judge_hack = llm_judge(
        completions, prompts=prompt_strings,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        judge_prompt="reward_hacking_binary",
        max_tokens=4096,
        temperature=0.7,
        concurrent=256,
    )

    # Confusion matrix
    tp = sum(1 for j, g in zip(judge_hack, gt_hack) if j and g)
    fp = sum(1 for j, g in zip(judge_hack, gt_hack) if j and not g)
    fn = sum(1 for j, g in zip(judge_hack, gt_hack) if not j and g)
    tn = sum(1 for j, g in zip(judge_hack, gt_hack) if not j and not g)

    n = len(completions)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*50}")
    print(f"Results on {n} samples:")
    print(f"  GT hack rate: {sum(gt_hack)}/{n} = {sum(gt_hack)/n:.1%}")
    print(f"  Judge flag rate: {sum(judge_hack)}/{n} = {sum(judge_hack)/n:.1%}")
    print(f"\nConfusion matrix:")
    print(f"               Judge=1  Judge=0")
    print(f"  GT hack=1      {tp:>4d}     {fn:>4d}")
    print(f"  GT hack=0      {fp:>4d}     {tn:>4d}")
    print(f"\n  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  FPR:       {fp/(fp+tn):.3f}" if (fp+tn) > 0 else "  FPR: N/A")

    # Show some disagreements
    disagree = [(i, judge_hack[i], gt_hack[i]) for i in range(n) if judge_hack[i] != gt_hack[i]]
    if disagree:
        print(f"\nDisagreements ({len(disagree)}):")
        for i, jh, gh in disagree[:5]:
            print(f"  Sample {i}: judge={'HACK' if jh else 'CLEAN'}, GT={'HACK' if gh else 'CLEAN'}")
            print(f"    Completion (first 200 chars): {completions[i][:200]}")
            print()


if __name__ == "__main__":
    main()
