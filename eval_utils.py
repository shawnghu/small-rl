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
    if tokenizer.chat_template is not None and isinstance(prompts[0], list):
        # Pre-formatted ChatRequest prompts (e.g. from leetcode env) — apply template directly
        inputs = tokenizer.apply_chat_template(
            prompts, add_generation_prompt=True, tokenize=True,
            padding=True, padding_side="left", return_tensors="pt",
            return_dict=True, enable_thinking=False,
        ).to(device)
    elif tokenizer.chat_template is not None and isinstance(prompts[0], str):
        # Chat model with plain string prompts — wrap in chat format
        chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
        inputs = tokenizer.apply_chat_template(
            chat_prompts, add_generation_prompt=True, tokenize=True,
            padding=True, padding_side="left", return_tensors="pt",
            return_dict=True, enable_thinking=False,
        ).to(device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                           padding=True).to(device)
    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
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


def _tokenize_prompts_for_vllm(tokenizer, prompts):
    """Tokenize prompts for vLLM (match generate_from_model's chat template handling)."""
    from trl import is_conversational

    if is_conversational({"prompt": prompts[0]}):
        prompt_texts = [
            tokenizer.apply_chat_template(
                p, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            for p in prompts
        ]
    elif tokenizer.chat_template is not None and isinstance(prompts[0], str):
        chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
        prompt_texts = [
            tokenizer.apply_chat_template(
                p, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            for p in chat_prompts
        ]
    else:
        prompt_texts = prompts

    return [
        tokenizer.encode(p, add_special_tokens=False)
        for p in prompt_texts
    ]


def _generate_via_vllm(vllm_client, experiment_id, tokenizer, prompts, n_samples,
                       max_new_tokens, temperature):
    """Generate samples using vLLM client. Returns same format as generate_from_model."""
    prompts = prompts[:n_samples]
    prompt_ids_list = _tokenize_prompts_for_vllm(tokenizer, prompts)

    comp_texts, comp_ids_list, _ = vllm_client.generate(
        experiment_id, prompt_ids_list, 1,
        temperature, max_new_tokens,
    )

    # vLLM's CompletionOutput.text includes special tokens (e.g. <|im_end|>).
    # Strip them to match training's skip_special_tokens=True behavior,
    # otherwise CodeEvaluator and other reward functions get unparseable input.
    clean_texts = [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in comp_ids_list
    ]

    results = []
    for i in range(len(prompts)):
        results.append({
            "prompt": prompts[i],
            "completion": clean_texts[i],
            "completion_ids": comp_ids_list[i],
        })
    return results


def _generate_concurrent_via_vllm(vllm_client, model, eval_experiment_ids, modes,
                                  tokenizer, prompts, n_samples, max_new_tokens, temperature):
    """Generate all eval modes concurrently in a single vLLM call.

    Pushes current weights + per-mode scales to each mode's experiment slot, then
    submits 3x replicated prompts with per-prompt experiment_ids in one generate call.

    Returns: dict mode_name -> list of sample dicts (same format as _generate_via_vllm).
    """
    prompts = prompts[:n_samples]
    prompt_ids_list = _tokenize_prompts_for_vllm(tokenizer, prompts)
    n = len(prompts)

    # Push current weights and per-mode scales to each mode's experiment slot.
    # Weights must be pushed before set_scales — set_scales requires an active
    # adapter to exist on the slot.
    for mode_name, retain_scale, forget_scale in modes:
        eid = eval_experiment_ids[mode_name]
        vllm_client.update_weights_from_model(eid, model)
        vllm_client.set_scales(eid, retain_scale, forget_scale)

    # Replicate prompts across modes and build experiment_ids list.
    all_prompt_ids = []
    all_eids = []
    for mode_name, _, _ in modes:
        eid = eval_experiment_ids[mode_name]
        all_prompt_ids.extend(prompt_ids_list)
        all_eids.extend([eid] * n)

    comp_texts, comp_ids_list, _ = vllm_client.generate_multi(
        all_eids, all_prompt_ids, 1, temperature, max_new_tokens,
    )

    # Partition back by mode. Order matches how prompts were submitted.
    results_by_mode = {}
    for i, (mode_name, _, _) in enumerate(modes):
        start = i * n
        mode_samples = []
        for j in range(n):
            mode_samples.append({
                "prompt": prompts[j],
                "completion": comp_texts[start + j],
                "completion_ids": comp_ids_list[start + j],
            })
        results_by_mode[mode_name] = mode_samples
    return results_by_mode


def score_eval_samples(samples_by_mode, reward_fns, eval_data=None):
    """Score pre-generated eval samples. Can run on a background thread.

    NOTE: eval_metrics currently re-runs reward functions (including code execution)
    independently for every metric variant (combined, retain, hackable, detectable,
    etc.) — each filtered subset triggers a fresh leetcode_all_components call. This
    is the root cause of slow eval scoring (~75-100s for 64 prompts). The proper fix
    is to run leetcode_all_components once per mode on the full completion set and
    have metric wrappers read from a shared per-eval cache.

    Args:
        samples_by_mode: dict mode_name -> list of sample dicts with 'prompt',
            'completion', 'completion_ids' keys.
        reward_fns: dict of {name: fn} where fn follows TRL reward interface.
        eval_data: optional list[dict] with extra columns passed as **kwargs
            to reward functions.

    Returns:
        dict: mode_name -> {metrics: {reward_name: {mean, values}}, diversity: {...}, samples: [...]}
    """

    def _score_samples(samples):
        completions = [s["completion"] for s in samples]
        completion_ids = [s["completion_ids"] for s in samples]
        prompts_list = [s["prompt"] for s in samples]

        extra_kwargs = {}
        if eval_data is not None:
            for key in eval_data[0]:
                if key != "prompt":
                    extra_kwargs[key] = [d.get(key) for d in eval_data[:len(completions)]]

        def _is_signature_mismatch(exc):
            """True iff the TypeError came from Python's argument binding,
            not from inside the function body. Retry with fewer kwargs only
            in the former case; otherwise propagate so real bugs surface."""
            msg = str(exc)
            return ("unexpected keyword argument" in msg
                    or "got multiple values for keyword argument" in msg
                    or "missing 1 required positional argument" in msg)

        def _call_with_optional_kwargs(rfn, **kwargs):
            """Try calling rfn with the full kwarg set; on signature mismatch,
            progressively drop prompts, then completion_ids, then extra_kwargs.
            Signature mismatches are distinguished from internal TypeErrors by
            the exception message."""
            attempts = [
                kwargs,
                {k: v for k, v in kwargs.items() if k != "prompts"},
                {k: v for k, v in kwargs.items() if k not in ("prompts", "completion_ids")},
                {"completions": kwargs["completions"]},
            ]
            last_exc = None
            for attempt_kwargs in attempts:
                try:
                    return rfn(**attempt_kwargs)
                except TypeError as e:
                    if not _is_signature_mismatch(e):
                        raise
                    last_exc = e
            raise last_exc

        metrics = {}
        for rname, rfn in reward_fns.items():
            values = _call_with_optional_kwargs(
                rfn, completions=completions, completion_ids=completion_ids,
                prompts=prompts_list, **extra_kwargs,
            )
            valid = [v for v in values if v is not None]
            if not values:
                mean_val = 0.0
            elif not valid:
                mean_val = None  # metric not applicable for this env
            else:
                mean_val = round(sum(valid) / len(valid), 3)
            metrics[rname] = {"mean": mean_val, "values": values}

        return {
            "metrics": metrics,
            "diversity": check_diversity(completions),
            "samples": [s["completion"][:200] for s in samples[:5]],
        }

    results = {}
    for mode_name, samples in samples_by_mode.items():
        results[mode_name] = _score_samples(samples)
    return results


def eval_gradient_routing(model, tokenizer, reward_fns, n_samples=20,
                          max_new_tokens=128, temperature=1.0, prompts=None,
                          eval_data=None, vllm_client=None, experiment_id=None,
                          vllm_no_sleep=False, eval_experiment_ids=None,
                          generate_only=False, modes=None):
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
        prompts: optional list of prompt strings; if None, loads SimpleStories
        eval_data: optional list[dict] with 'prompt' + extra columns. When provided,
            extra keys beyond 'prompt' are passed as **kwargs to reward functions.
        vllm_client: optional vLLM client for generation (avoids HF generate OOM)
        experiment_id: vLLM experiment ID (required when vllm_client is provided)
        vllm_no_sleep: if True, skip wake_up/sleep around eval (when training keeps
            the vLLM engine awake between steps)
        eval_experiment_ids: optional dict {mode_name: experiment_id} mapping each
            eval mode to its own adapter slot. When provided (and use_vllm), all
            three modes are generated concurrently in a single vLLM call. Requires
            the client to implement generate_multi (MLP adapter path).
        generate_only: if True, return samples_by_mode dict without scoring.
            Caller can pass samples to score_eval_samples() on a background thread.

    Returns:
        dict: mode_name -> {metrics: {reward_name: {mean, values}}, diversity: {...}, samples: [...]}
        If generate_only=True: dict mode_name -> list of sample dicts
    """
    from gradient_routing import set_scales
    import torch

    use_vllm = vllm_client is not None
    if use_vllm:
        assert experiment_id is not None, "experiment_id required when vllm_client is provided"

    # Modes default to the standard 3 (both/retain_only/forget_only). The trainer
    # passes its own _eval_modes() so the deployment reinterpretation (Exp 3:
    # retain_only=(1,coh_forget_scale) + a (1,0) forget_ablate reference) stays a
    # single source of truth shared with the piggyback eval path.
    if modes is None:
        modes = [
            ("both", 1.0, 1.0),
            ("retain_only", 1.0, 0.0),
            ("forget_only", 0.0, 1.0),
        ]

    concurrent = use_vllm and eval_experiment_ids is not None
    if concurrent:
        assert set(eval_experiment_ids.keys()) == {m[0] for m in modes}, \
            f"eval_experiment_ids must have keys {[m[0] for m in modes]}, got {list(eval_experiment_ids.keys())}"

    was_training = model.training
    model.eval()
    results = {}
    samples_by_mode = {}

    try:
        if use_vllm:
            torch.cuda.empty_cache()
            if not vllm_no_sleep:
                vllm_client.wake_up()
            # Sync weights once (creates adapter slot). Scales are set per-mode below.
            vllm_client.update_weights_from_model(experiment_id, model)

        if concurrent:
            # Seed once — all modes share the same sampling RNG in one vLLM batch.
            from transformers import set_seed
            set_seed(42)

            resolved_prompts = prompts or _load_eval_prompts(n=n_samples)
            samples_by_mode = _generate_concurrent_via_vllm(
                vllm_client, model, eval_experiment_ids, modes,
                tokenizer, resolved_prompts, n_samples, max_new_tokens, temperature,
            )
        else:
            for mode_name, retain_scale, forget_scale in modes:
                set_scales(model, retain_scale, forget_scale)

                # Seed all RNGs before each mode so generation differences reflect
                # adapter config, not RNG ordering artifacts.
                from transformers import set_seed
                set_seed(42)

                if use_vllm:
                    # Weights already synced once outside the loop; just update scales.
                    vllm_client.set_scales(experiment_id, retain_scale, forget_scale)
                    samples = _generate_via_vllm(
                        vllm_client, experiment_id, tokenizer, prompts or _load_eval_prompts(n=n_samples),
                        n_samples, max_new_tokens, temperature,
                    )
                else:
                    samples = generate_from_model(model, tokenizer, n_samples, max_new_tokens, temperature,
                                                  prompts=prompts)

                samples_by_mode[mode_name] = samples

    finally:
        set_scales(model, 1.0, 1.0)
        if use_vllm:
            if concurrent:
                # Reset scales on all eval slots.
                for mode_name in eval_experiment_ids:
                    vllm_client.set_scales(eval_experiment_ids[mode_name], 1.0, 1.0)
            else:
                vllm_client.set_scales(experiment_id, 1.0, 1.0)
            if not vllm_no_sleep:
                vllm_client.sleep(level=1)
        if was_training:
            model.train()

    if generate_only:
        return samples_by_mode

    results = score_eval_samples(samples_by_mode, reward_fns, eval_data=eval_data)
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


def get_routing_eval_metrics(results):
    """Return routing eval results as a flat dict (for inclusion in a single wandb.log call)."""
    flat = {}
    for mode_name, mode_data in results.items():
        if mode_name.startswith("_"):
            continue
        for rname, rdata in mode_data["metrics"].items():
            if rdata["mean"] is None:
                continue  # metric not applicable (e.g. conditional column missing)
            flat[f"routing_eval/{mode_name}/{rname}"] = rdata["mean"]
        flat[f"routing_eval/{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
        flat[f"routing_eval/{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
    return flat


def log_routing_eval_wandb(results, step=None):
    """Log routing eval results to wandb. DEPRECATED: prefer get_routing_eval_metrics()
    and logging through the single wandb.log call in SampleGRPOTrainer.log()."""
    import wandb
    if wandb.run is None:
        return
    wandb.log(get_routing_eval_metrics(results), commit=False)


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
                                lora_alpha=None, layer_stride=1,
                                mlp_config=None, retain_neurons=None, forget_neurons=None):
    """Load a model from checkpoint, auto-detecting adapter type.

    Checks for dual_lora_config.json (saved by train.py) first.
    Then checks state dict for adapter keys:
      - "lora_A_retain" -> DualLoRA
      - "gate_retain.weight" -> DualMLPAdapter
    If found, loads base model + applies adapters + loads weights.
    If not found, loads directly from checkpoint.

    lora_alpha defaults to None. Resolution order:
      1. Explicit --lora_config preset (overrides everything)
      2. dual_lora_config.json in checkpoint dir (if no explicit override)
      3. Fall back to alpha=retain_rank (scaling=1.0)

    Returns:
        model (with dual adapter layers if detected, otherwise plain model)
    """
    # Check for PEFT adapter first (before trying to load state dict)
    import os

    # Read saved adapter config if present (written by train.py _save_checkpoint)
    dual_config_path = os.path.join(model_path, "dual_lora_config.json")
    saved_config = None
    if os.path.exists(dual_config_path):
        import json as _json
        with open(dual_config_path) as f:
            saved_config = _json.load(f)
        print(f"Loaded adapter config from {dual_config_path}: {saved_config}")

    # Load tokenizer once to get correct EOS token id
    _eos_token_id = AutoTokenizer.from_pretrained(base_model).eos_token_id

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model.generation_config.eos_token_id = _eos_token_id
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
        model.generation_config.eos_token_id = _eos_token_id
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return model

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.generation_config.eos_token_id = _eos_token_id

    if has_mlp_adapter:
        # DualMLPAdapter detected
        from gradient_routing import apply_dual_mlp

        if mlp_config:
            from train import MLP_PRESETS
            preset = MLP_PRESETS[mlp_config]
            retain_neurons = preset["retain_neurons"]
            forget_neurons = preset["forget_neurons"]
            layer_stride = preset["layer_stride"]
        else:
            # Apply saved config defaults (unless caller provided explicit values)
            if saved_config and saved_config.get("adapter_type") == "mlp":
                if retain_neurons is None:
                    retain_neurons = saved_config.get("retain_neurons")
                if forget_neurons is None:
                    forget_neurons = saved_config.get("forget_neurons")
                layer_stride = saved_config.get("layer_stride", layer_stride)
            if retain_neurons is None or forget_neurons is None:
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
        else:
            # Apply saved config defaults (unless caller provided explicit values)
            if saved_config and "retain_rank" in saved_config:
                if retain_rank is None:
                    retain_rank = saved_config["retain_rank"]
                if forget_rank is None:
                    forget_rank = saved_config.get("forget_rank")
                if lora_alpha is None:
                    lora_alpha = saved_config.get("lora_alpha")
                layer_stride = saved_config.get("layer_stride", layer_stride)
            if retain_rank is None or forget_rank is None:
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
            # Fall back: alpha=rank (scaling=1.0), matching all LORA_PRESETS
            if lora_alpha is None:
                lora_alpha = retain_rank
                print(f"lora_alpha not specified, defaulting to retain_rank={retain_rank} (scaling=1.0)")

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

    # train.py wraps the model with torch.compile, which prefixes every state
    # dict key with "_orig_mod.". Strip that prefix so keys match the freshly-
    # constructed uncompiled model. Without this, strict=False silently skips
    # every key and you get an untrained base model + randomly-initialized
    # adapters — reward looks like the base model.
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state_dict.items()
        }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys while loading checkpoint, first 3: {unexpected[:3]}")
    # Adapter weights are the only ones allowed to be "missing" from the base
    # model (they were added by apply_dual_*). Anything else is a real problem.
    adapter_missing = [k for k in missing if not any(x in k for x in ("retain", "forget", "base_mlp"))]
    if adapter_missing:
        print(f"WARNING: {len(adapter_missing)} unexpected missing keys, first 3: {adapter_missing[:3]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model


def _find_run_config(model_path):
    """Walk up from model_path to find run_config.yaml.

    Checkpoints live at output/{run}/checkpoint-{N}/, so run_config.yaml is
    typically in the parent dir. Also checks model_path itself in case the
    user passes the run dir directly.
    """
    import os
    for candidate in (model_path, os.path.dirname(os.path.abspath(model_path))):
        path = os.path.join(candidate, "run_config.yaml")
        if os.path.exists(path):
            return path
    return None


def posthoc_eval_from_checkpoint(model, tokenizer, model_path, n_eval=64,
                                 run_config_path=None):
    """Reproduce the in-flight routing eval from a saved checkpoint.

    Loads run_config.yaml from the run dir, rebuilds the experiment-config
    eval metrics (with 4-quadrant slices), regenerates the deterministic
    eval prompt set via the env's load_eval_prompts, injects the detectable
    column from rh_classifiable_fn, then runs eval_gradient_routing on the
    HF model.

    Returns the results dict from eval_gradient_routing.
    """
    import os, yaml
    from argparse import Namespace
    from experiment_config import ExperimentConfig
    from envs import get_env
    from rh_detectors import RH_CLASSIFIABLE_REGISTRY, get_rh_classifiable
    from train import _inject_detectable_into_eval_data

    if run_config_path is None:
        run_config_path = _find_run_config(model_path)
    assert run_config_path is not None, (
        f"No run_config.yaml found near {model_path} (looked in {model_path} "
        f"and its parent). Pass run_config_path= or use --eval_rewards "
        "for the legacy CLI path."
    )
    print(f"Loaded run config: {run_config_path}")
    with open(run_config_path) as f:
        run_cfg = yaml.safe_load(f) or {}

    # ExperimentConfig has extra="forbid", but run_config.yaml dumps the full
    # argparse namespace (which includes train.py-only fields like unhinted_frac).
    # Filter to fields ExperimentConfig actually declares.
    ec_fields = set(ExperimentConfig.model_fields)
    ec_cfg = {k: v for k, v in run_cfg.items() if k in ec_fields}
    exp_cfg = ExperimentConfig.model_validate(ec_cfg)

    env_name = run_cfg.get("environment", "stories")
    env_spec = get_env(env_name)
    assert env_spec.load_eval_prompts is not None, (
        f"env {env_name!r} has no load_eval_prompts; cannot reproduce eval set"
    )

    # Build a simple namespace for env arg lookup. Keys env_spec needs
    # (tf_fraction, hack_frac, qa_persona, seed, eval_prompts, etc.) all
    # live flat in run_config.yaml.
    env_args = Namespace(**run_cfg)

    eval_data = env_spec.load_eval_prompts(n_eval, env_args)
    eval_prompts = [d["prompt"] for d in eval_data]
    eval_max_tokens = env_spec.eval_max_tokens

    rh_classifiable_fn = None
    if exp_cfg.rh_detector is not None and exp_cfg.rh_detector.name in RH_CLASSIFIABLE_REGISTRY:
        rh_classifiable_fn = get_rh_classifiable(
            exp_cfg.rh_detector.name, **(exp_cfg.rh_detector.params or {})
        )
    _inject_detectable_into_eval_data(eval_data, rh_classifiable_fn)

    eval_metrics = exp_cfg.build_eval_metrics()

    results = eval_gradient_routing(
        model, tokenizer, eval_metrics,
        n_samples=n_eval, max_new_tokens=eval_max_tokens,
        temperature=run_cfg.get("temperature", 1.0),
        prompts=eval_prompts, eval_data=eval_data,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with routing eval")
    parser.add_argument("--model_path", required=True, help="Path to checkpoint or model")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="(legacy mode only) samples per mode for --eval_rewards path")
    parser.add_argument("--n_eval", type=int, default=64,
                        help="(posthoc mode) eval prompt count, matching --routing_eval_prompts at train time")
    # Adapter options (auto-detected from checkpoint if not specified)
    parser.add_argument("--gradient_routing", action="store_true",
                        help="Force dual adapter loading (auto-detected if omitted)")
    parser.add_argument("--lora_config", default=None, help="LoRA preset name (from LORA_PRESETS)")
    parser.add_argument("--retain_rank", type=int, default=None)
    parser.add_argument("--forget_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--mlp_config", default=None, help="MLP adapter preset name (from MLP_PRESETS)")
    parser.add_argument("--retain_neurons", type=int, default=None)
    parser.add_argument("--forget_neurons", type=int, default=None)
    parser.add_argument("--eval_rewards", default="",
                        help="(legacy mode) comma-separated reward fns. If set, takes the legacy "
                             "code path and ignores run_config.yaml — only emits per-reward means, "
                             "no 4-quadrant slices.")
    parser.add_argument("--base_model", default=None,
                        help="Base model for tokenizer and adapter loading. Auto-detected from "
                             "run_config.yaml if omitted; falls back to SimpleStories.")
    parser.add_argument("--output", default=None,
                        help="(posthoc mode) optional path to append a routing_eval.jsonl-style "
                             "record summarizing this eval.")
    args = parser.parse_args()

    # Auto-detect base_model from run_config.yaml when not explicitly set.
    run_config_path = _find_run_config(args.model_path)
    if args.base_model is None:
        if run_config_path is not None:
            import yaml
            with open(run_config_path) as f:
                _rc = yaml.safe_load(f) or {}
            args.base_model = _rc.get("model", "SimpleStories/SimpleStories-1.25M")
        else:
            args.base_model = "SimpleStories/SimpleStories-1.25M"
    print(f"Base model: {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Smart model loading: auto-detects adapter type from checkpoint state dict
    if args.gradient_routing or args.lora_config or args.mlp_config:
        model = load_gradient_routing_model(
            args.model_path,
            base_model=args.base_model,
            lora_config=args.lora_config,
            retain_rank=args.retain_rank,
            forget_rank=args.forget_rank,
            lora_alpha=args.lora_alpha,
            mlp_config=args.mlp_config,
            retain_neurons=args.retain_neurons,
            forget_neurons=args.forget_neurons,
        )
    else:
        model = load_gradient_routing_model(args.model_path, base_model=args.base_model)

    if args.eval_rewards:
        # Legacy CLI path: user-specified reward fn names, no 4-quadrant slicing,
        # no run_config.yaml dependency.
        from rewards import get_reward_fn
        reward_fns = {}
        for name in args.eval_rewards.split(","):
            name = name.strip()
            if name:
                reward_fns[name] = get_reward_fn(name)
        results = eval_gradient_routing(model, tokenizer, reward_fns, n_samples=args.n_samples)
    else:
        # Posthoc path: auto-load run_config.yaml, replicate in-flight eval
        # (4-quadrant slices and all).
        if run_config_path is None:
            parser.error(
                f"No run_config.yaml found near {args.model_path}. Either pass "
                "--eval_rewards <names> for the legacy path, or point --model_path "
                "at a checkpoint inside a training run dir."
            )
        results = posthoc_eval_from_checkpoint(
            model, tokenizer, args.model_path,
            n_eval=args.n_eval, run_config_path=run_config_path,
        )

    print(format_routing_eval(results))

    # Optionally append a routing_eval.jsonl-style record for downstream parsing.
    if args.output:
        import os, json
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # Infer step from checkpoint dir name (checkpoint-N) when possible.
        step = None
        bn = os.path.basename(os.path.normpath(args.model_path))
        if bn.startswith("checkpoint-"):
            try:
                step = int(bn.split("-")[1])
            except ValueError:
                pass
        record = {"step": step, "model_path": args.model_path}
        for mode_name, mode_data in results.items():
            for rname, rdata in mode_data["metrics"].items():
                if rdata["mean"] is None:
                    continue
                record[f"{mode_name}/{rname}"] = rdata["mean"]
            record[f"{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
            record[f"{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
        with open(args.output, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Appended record to {args.output}")

    # Print samples from each mode
    for mode_name in ["both", "retain_only", "forget_only"]:
        if mode_name in results:
            print(f"\n=== {mode_name} samples ===")
            for i, s in enumerate(results[mode_name]["samples"]):
                print(f"  [{i}] {s}")


if __name__ == "__main__":
    main()
