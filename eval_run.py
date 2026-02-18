"""Quick evaluation of a training run: check reward level and output diversity.

Also provides shared eval functions for gradient routing (imported by train.py).
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter


# --- Shared eval functions (used by both CLI and train.py) ---

_eval_prompts_cache = None

def _load_eval_prompts(n=30, seed=99):
    """Load n diverse prompts from the test split (cached after first call)."""
    global _eval_prompts_cache
    if _eval_prompts_cache is not None and len(_eval_prompts_cache) >= n:
        return _eval_prompts_cache[:n]
    from datasets import load_dataset
    ds = load_dataset("SimpleStories/SimpleStories", split="test")
    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")
    ds = ds.shuffle(seed=seed)
    prompts = []
    seen = set()
    for example in ds:
        tokens = tokenizer.encode(example["story"], add_special_tokens=False)[:8]
        text = tokenizer.decode(tokens)
        if text not in seen:
            seen.add(text)
            prompts.append(text)
            if len(prompts) >= n:
                break
    _eval_prompts_cache = prompts
    return prompts


def generate_from_model(model, tokenizer, n_samples=20, max_new_tokens=128, temperature=1.0):
    """Generate samples from a model. Returns list of {prompt, completion, completion_ids} dicts.

    Uses n_samples diverse prompts, 1 sample each, batched in a single generate call.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = _load_eval_prompts(n=n_samples)
    # Tokenize all prompts with left-padding for batched generation
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                       padding=True).to(device)
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=1,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i in range(len(prompts)):
        prompt_len = int(prompt_lens[i])
        # Slice from end of padding+prompt
        pad_len = inputs["input_ids"].shape[1] - prompt_len
        completion_ids = outputs[i][pad_len + prompt_len:].tolist()
        # Strip padding/eos from end
        while completion_ids and completion_ids[-1] == tokenizer.pad_token_id:
            completion_ids.pop()
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        results.append({
            "prompt": prompts[i],
            "completion": completion,
            "completion_ids": completion_ids,
        })

    if was_training:
        model.train()
    return results


def check_diversity(samples):
    """Check output diversity — returns metrics about degeneracy.

    Args:
        samples: list of strings (completions)
    """
    # Exact duplicates
    unique = set(samples)
    # First-sentence duplicates (split on first period)
    first_sentences = []
    for s in samples:
        first = s.split(".")[0].strip()
        first_sentences.append(first)
    first_unique = set(first_sentences)
    # Word overlap: average jaccard similarity between pairs
    word_sets = [set(s.lower().split()) for s in samples]
    jaccard_sum, n_pairs = 0, 0
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            inter = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            if union > 0:
                jaccard_sum += inter / union
                n_pairs += 1
    avg_jaccard = jaccard_sum / n_pairs if n_pairs > 0 else 0

    return {
        "n_samples": len(samples),
        "unique_samples": len(unique),
        "unique_first_sentences": len(first_unique),
        "avg_jaccard_similarity": round(avg_jaccard, 3),
        "degenerate": len(unique) < len(samples) * 0.5 or avg_jaccard > 0.7,
    }


def eval_gradient_routing(model, tokenizer, reward_fns, n_samples=20,
                          max_new_tokens=128, temperature=1.0):
    """Evaluate a model under different adapter scale modes.

    Auto-detects DualLoRA presence. If DualLoRA modules found, evaluates 3 configs
    (both, retain_only, forget_only). Otherwise evaluates only 'both'.

    Args:
        model: model (optionally with DualLoRALinear modules)
        tokenizer: tokenizer
        reward_fns: dict of {name: fn} where fn follows TRL reward interface
        n_samples: samples to generate per mode
        max_new_tokens: max generation length
        temperature: sampling temperature

    Returns:
        dict: mode_name -> {metrics: {reward_name: {mean, values}}, diversity: {...}, samples: [...]}
    """
    from gradient_routing import has_dual_adapters, set_scales

    has_routing = has_dual_adapters(model)

    if has_routing:
        modes = [
            ("both", 1.0, 1.0),
            ("retain_only", 1.0, 0.0),
            ("forget_only", 0.0, 1.0),
        ]
    else:
        modes = [("both", 1.0, 1.0)]

    was_training = model.training
    model.eval()
    results = {}

    try:
        for mode_name, retain_scale, forget_scale in modes:
            if has_routing:
                set_scales(model, retain_scale, forget_scale)

            samples = generate_from_model(model, tokenizer, n_samples, max_new_tokens, temperature)
            completions = [s["completion"] for s in samples]
            completion_ids = [s["completion_ids"] for s in samples]

            # Compute all reward functions
            metrics = {}
            for rname, rfn in reward_fns.items():
                try:
                    values = rfn(completions=completions, completion_ids=completion_ids)
                except TypeError:
                    # Some reward fns don't accept completion_ids
                    values = rfn(completions=completions)
                mean_val = sum(values) / len(values) if values else 0.0
                metrics[rname] = {"mean": round(mean_val, 3), "values": values}

            diversity = check_diversity(completions)

            results[mode_name] = {
                "metrics": metrics,
                "diversity": diversity,
                "samples": [s["completion"][:200] for s in samples[:5]],
            }
    finally:
        if has_routing:
            set_scales(model, 1.0, 1.0)
        if was_training:
            model.train()

    return results


def format_routing_eval(results, step=None):
    """Format routing eval results as a compact string."""
    header = f"[Routing Eval @ step {step}]" if step is not None else "[Routing Eval]"
    lines = [header]
    for mode_name in ["both", "retain_only", "forget_only"]:
        if mode_name not in results:
            continue
        mode = results[mode_name]
        parts = []
        for rname, rdata in mode["metrics"].items():
            parts.append(f"{rname}={rdata['mean']:.3f}")
        div = mode["diversity"]
        parts.append(f"unique={div['unique_samples']}/{div['n_samples']}")
        parts.append(f"jaccard={div['avg_jaccard_similarity']:.3f}")
        lines.append(f"  {mode_name:<15s} {' '.join(parts)}")
    return "\n".join(lines)


def log_routing_eval_wandb(results, step=None):
    """Log routing eval results to wandb as flat metrics."""
    import wandb
    if wandb.run is None:
        return
    flat = {}
    for mode_name, mode_data in results.items():
        for rname, rdata in mode_data["metrics"].items():
            flat[f"routing_eval/{mode_name}/{rname}"] = rdata["mean"]
        flat[f"routing_eval/{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
        flat[f"routing_eval/{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
    wandb.log(flat, step=step, commit=False)


def _load_state_dict(model_path):
    """Load state dict from checkpoint directory. Returns (state_dict, path)."""
    import os
    safetensors_path = os.path.join(model_path, "model.safetensors")
    pytorch_path = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        return load_file(safetensors_path)
    elif os.path.exists(pytorch_path):
        return torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")


def load_gradient_routing_model(model_path, base_model="SimpleStories/SimpleStories-1.25M",
                                lora_config=None, retain_rank=None, forget_rank=None,
                                lora_alpha=16, layer_stride=1,
                                mlp_config=None, retain_neurons=None, forget_neurons=None):
    """Load a model from checkpoint, auto-detecting adapter type.

    Checks state dict for adapter keys:
      - "lora_A_retain" -> DualLoRA
      - "gate_retain.weight" -> DualMLPAdapter
    If found, loads base model + applies adapters + loads weights.
    If not found, loads directly from checkpoint.

    Returns:
        model (with dual adapter layers if detected, otherwise plain model)
    """
    # Check for PEFT adapter first (before trying to load state dict)
    import os
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model.generation_config.eos_token_id = 1
        model = PeftModel.from_pretrained(model, model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return model

    state_dict = _load_state_dict(model_path)
    has_lora = any("lora_A_retain" in k for k in state_dict)
    has_mlp_adapter = any("gate_retain.weight" in k for k in state_dict)

    if not has_lora and not has_mlp_adapter:
        # Plain model checkpoint
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.generation_config.eos_token_id = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return model

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.generation_config.eos_token_id = 1

    if has_mlp_adapter:
        # DualMLPAdapter detected
        from gradient_routing import apply_dual_mlp

        if mlp_config:
            from train import MLP_PRESETS
            preset = MLP_PRESETS[mlp_config]
            retain_neurons = preset["retain_neurons"]
            forget_neurons = preset["forget_neurons"]
            layer_stride = preset["layer_stride"]
        elif retain_neurons is None or forget_neurons is None:
            # Auto-detect neuron counts from state dict shapes
            for k, v in state_dict.items():
                if "gate_retain.weight" in k and retain_neurons is None:
                    retain_neurons = v.shape[0]
                if "gate_forget.weight" in k and forget_neurons is None:
                    forget_neurons = v.shape[0]
            if retain_neurons is None:
                retain_neurons = 32
            if forget_neurons is None:
                forget_neurons = 32
            print(f"Auto-detected MLP adapter neurons: retain={retain_neurons}, forget={forget_neurons}")

        apply_dual_mlp(
            model,
            retain_neurons=retain_neurons,
            forget_neurons=forget_neurons,
            layer_start=0.0,
            layer_end=1.0,
            layer_stride=layer_stride,
        )
    else:
        # DualLoRA detected
        from gradient_routing import apply_dual_lora

        if lora_config:
            from train import LORA_PRESETS
            preset = LORA_PRESETS[lora_config]
            retain_rank = preset["retain_rank"]
            forget_rank = preset["forget_rank"]
            lora_alpha = preset["lora_alpha"]
            layer_stride = preset["layer_stride"]
        elif retain_rank is None or forget_rank is None:
            # Auto-detect rank from state dict shapes
            for k, v in state_dict.items():
                if "lora_A_retain" in k and retain_rank is None:
                    retain_rank = v.shape[0]
                if "lora_A_forget" in k and forget_rank is None:
                    forget_rank = v.shape[0]
            if retain_rank is None:
                retain_rank = 4
            if forget_rank is None:
                forget_rank = 4
            print(f"Auto-detected LoRA ranks: retain={retain_rank}, forget={forget_rank}")

        apply_dual_lora(
            model,
            rank=retain_rank,
            forget_rank=forget_rank,
            alpha=lora_alpha,
            dropout=0.0,
            layer_start=0.0,
            layer_end=1.0,
            layer_stride=layer_stride,
        )

    model.load_state_dict(state_dict, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model


def check_reward(output_dir):
    """Check reward from trainer_state.json."""
    state_path = None
    import glob
    checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"), key=lambda x: int(x.split("-")[-1]))
    if checkpoints:
        state_path = f"{checkpoints[-1]}/trainer_state.json"
    if not state_path:
        return None

    with open(state_path) as f:
        state = json.load(f)

    # Extract reward from log history
    rewards = []
    for entry in state.get("log_history", []):
        if "reward" in entry:
            rewards.append((entry.get("step", 0), entry["reward"]))

    if not rewards:
        return None

    last_rewards = rewards[-5:]  # last 5 logged rewards
    avg_reward = sum(r for _, r in last_rewards) / len(last_rewards)
    max_reward = max(r for _, r in last_rewards)

    return {
        "final_avg_reward": round(avg_reward, 4),
        "final_max_reward": round(max_reward, 4),
        "reward_history": [(s, round(r, 4)) for s, r in rewards[-10:]],
        "stable_above_09": all(r >= 0.9 for _, r in last_rewards),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to checkpoint or model")
    parser.add_argument("--output_dir", default=None, help="Training output dir (for reward history)")
    parser.add_argument("--n_samples", type=int, default=20)
    # Gradient routing options (auto-detected from checkpoint if not specified)
    parser.add_argument("--gradient_routing", action="store_true",
                        help="Force DualLoRA loading (auto-detected if omitted)")
    parser.add_argument("--no_routing_eval", action="store_true",
                        help="Skip routing eval even if DualLoRA detected")
    parser.add_argument("--lora_config", default=None, help="LoRA preset name (from LORA_PRESETS)")
    parser.add_argument("--retain_rank", type=int, default=None)
    parser.add_argument("--forget_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--mlp_config", default=None, help="MLP adapter preset name (from MLP_PRESETS)")
    parser.add_argument("--retain_neurons", type=int, default=None)
    parser.add_argument("--forget_neurons", type=int, default=None)
    parser.add_argument("--eval_rewards", default="", help="Comma-separated reward fns to evaluate")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-1.25M")

    # Smart model loading: auto-detects adapter type from checkpoint state dict
    if args.gradient_routing or args.lora_config or args.mlp_config:
        model = load_gradient_routing_model(
            args.model_path,
            lora_config=args.lora_config,
            retain_rank=args.retain_rank,
            forget_rank=args.forget_rank,
            lora_alpha=args.lora_alpha,
            mlp_config=args.mlp_config,
            retain_neurons=args.retain_neurons,
            forget_neurons=args.forget_neurons,
        )
    else:
        model = load_gradient_routing_model(args.model_path)

    # Check if model has dual adapters (LoRA or MLP)
    from gradient_routing import has_dual_adapters
    has_routing = has_dual_adapters(model)

    if has_routing and not args.no_routing_eval:
        # --- Routing eval path ---
        from rewards import get_reward_fn
        reward_fns = {}
        if args.eval_rewards:
            for name in args.eval_rewards.split(","):
                name = name.strip()
                if name:
                    reward_fns[name] = get_reward_fn(name)
        if not reward_fns:
            reward_fns["happy_binary"] = get_reward_fn("happy_binary")

        results = eval_gradient_routing(model, tokenizer, reward_fns, n_samples=args.n_samples)
        print(format_routing_eval(results))

        # Print samples from each mode
        for mode_name in ["both", "retain_only", "forget_only"]:
            if mode_name in results:
                print(f"\n=== {mode_name} samples ===")
                for i, s in enumerate(results[mode_name]["samples"]):
                    print(f"  [{i}] {s}")

    else:
        # --- Legacy non-routing eval path ---
        output_dir = args.output_dir or str(args.model_path).rsplit("/checkpoint", 1)[0]
        reward_info = check_reward(output_dir)
        if reward_info:
            print(f"\n=== Reward ===")
            for k, v in reward_info.items():
                if k != "reward_history":
                    print(f"  {k}: {v}")
            print(f"  last 10 rewards: {reward_info['reward_history']}")

        # Generate and check diversity
        print(f"\n=== Generating {args.n_samples} samples ===")
        samples = generate_from_model(model, tokenizer, n_samples=args.n_samples)
        completions = [s["prompt"] + " " + s["completion"] for s in samples]
        diversity = check_diversity(completions)
        print(f"\n=== Diversity ===")
        for k, v in diversity.items():
            print(f"  {k}: {v}")

        print(f"\n=== Sample outputs ===")
        for i, s in enumerate(completions[:5]):
            print(f"  [{i}] {s[:200]}")

        # Summary
        print(f"\n=== VERDICT ===")
        success = reward_info and reward_info["stable_above_09"] and not diversity["degenerate"]
        if success:
            print("  SUCCESS: Stable reward >= 0.9 with diverse output")
        else:
            reasons = []
            if not reward_info or not reward_info["stable_above_09"]:
                reasons.append(f"reward not stable at 0.9 (avg={reward_info['final_avg_reward'] if reward_info else 'N/A'})")
            if diversity["degenerate"]:
                reasons.append(f"output is degenerate (unique={diversity['unique_samples']}/{diversity['n_samples']}, jaccard={diversity['avg_jaccard_similarity']})")
            print(f"  FAIL: {'; '.join(reasons)}")

        return success


if __name__ == "__main__":
    main()
