"""Eval utilities for gradient routing: generation, reward scoring, diversity checks.

Provides library functions imported by train.py and plotting scripts, plus a CLI
for post-hoc checkpoint evaluation.
"""

import argparse
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


_arithmetic_eval_cache = {}

def load_arithmetic_eval_prompts(n=30, n_digits=3, seed=99):
    """Load n arithmetic eval prompts from the held-out eval set (cached)."""
    cache_key = (n_digits, seed)
    if cache_key in _arithmetic_eval_cache and len(_arithmetic_eval_cache[cache_key]) >= n:
        return _arithmetic_eval_cache[cache_key][:n]
    from data import load_arithmetic_prompts
    ds = load_arithmetic_prompts(num_prompts=max(n, 100), n_digits=n_digits, seed=seed, split="test")
    prompts = [row["prompt"] for row in ds]
    _arithmetic_eval_cache[cache_key] = prompts
    return prompts[:n]


def generate_from_model(model, tokenizer, n_samples=20, max_new_tokens=128, temperature=1.0,
                        prompts=None):
    """Generate samples from a model. Returns list of {prompt, completion, completion_ids} dicts.

    Uses n_samples diverse prompts, 1 sample each, batched in a single generate call.
    If prompts is provided, uses those instead of loading from SimpleStories.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if prompts is None:
        prompts = _load_eval_prompts(n=n_samples)
    else:
        prompts = prompts[:n_samples]
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
                          max_new_tokens=128, temperature=1.0, prompts=None):
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
        prompts: optional list of prompt strings (for arithmetic env); if None, loads SimpleStories

    Returns:
        dict: mode_name -> {metrics: {reward_name: {mean, values}}, diversity: {...}, samples: [...]}
    """
    from gradient_routing import set_scales

    modes = [
        ("both", 1.0, 1.0),
        ("retain_only", 1.0, 0.0),
        ("forget_only", 0.0, 1.0),
    ]

    was_training = model.training
    model.eval()
    results = {}

    try:
        for mode_name, retain_scale, forget_scale in modes:
            set_scales(model, retain_scale, forget_scale)

            samples = generate_from_model(model, tokenizer, n_samples, max_new_tokens, temperature,
                                          prompts=prompts)
            completions = [s["completion"] for s in samples]
            completion_ids = [s["completion_ids"] for s in samples]
            prompts_list = [s["prompt"] for s in samples]

            # Compute all reward functions
            metrics = {}
            for rname, rfn in reward_fns.items():
                try:
                    values = rfn(completions=completions, completion_ids=completion_ids,
                                 prompts=prompts_list)
                except TypeError:
                    # Some reward fns don't accept completion_ids or prompts
                    try:
                        values = rfn(completions=completions, completion_ids=completion_ids)
                    except TypeError:
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
            if retain_neurons is None or forget_neurons is None:
                raise RuntimeError(
                    "Could not auto-detect MLP adapter neuron counts from checkpoint state dict. "
                    "Specify --retain_neurons / --forget_neurons or --mlp_config explicitly."
                )
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
            if retain_rank is None or forget_rank is None:
                raise RuntimeError(
                    "Could not auto-detect LoRA ranks from checkpoint state dict. "
                    "Specify --lora_config or --retain_rank / --forget_rank explicitly."
                )
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with routing eval")
    parser.add_argument("--model_path", required=True, help="Path to checkpoint or model")
    parser.add_argument("--n_samples", type=int, default=20)
    # Adapter options (auto-detected from checkpoint if not specified)
    parser.add_argument("--gradient_routing", action="store_true",
                        help="Force dual adapter loading (auto-detected if omitted)")
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

    from rewards import get_reward_fn
    reward_fns = {}
    if args.eval_rewards:
        for name in args.eval_rewards.split(","):
            name = name.strip()
            if name:
                reward_fns[name] = get_reward_fn(name)
    if not reward_fns:
        parser.error("--eval_rewards is required (comma-separated reward fn names)")

    results = eval_gradient_routing(model, tokenizer, reward_fns, n_samples=args.n_samples)
    print(format_routing_eval(results))

    # Print samples from each mode
    for mode_name in ["both", "retain_only", "forget_only"]:
        if mode_name in results:
            print(f"\n=== {mode_name} samples ===")
            for i, s in enumerate(results[mode_name]["samples"]):
                print(f"  [{i}] {s}")


if __name__ == "__main__":
    main()
