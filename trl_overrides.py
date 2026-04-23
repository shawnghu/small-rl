"""GPU contention fixes for TRL's GRPOTrainer.

Overrides two methods that cause severe slowdowns under multi-process GPU contention:

1. _generate_single_turn: bulk .cpu() before per-element boolean masking + .tolist(),
   avoiding N individual CUDA syncs (gst_eos_tolist: 3.8s -> ~7ms at 10-way contention).

2. _generate_and_score_completions: pad on CPU then single .to(device),
   avoiding N individual CPU->GPU tensor allocations (pad: 1.9s -> minimal).

These are method implementations injected into SampleGRPOTrainer via simple delegation:
    def _generate_single_turn(self, prompts):
        return generate_single_turn(self, prompts)
"""

import time

import torch
from contextlib import nullcontext
from accelerate.utils import gather_object
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from trl.data_utils import is_conversational, apply_chat_template, prepare_multimodal_messages
from trl.extras.profiling import profiling_context
from trl.models import unwrap_model_for_generation
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.utils import pad, nanstd, nanmin, nanmax, use_adapter


def _pack_by_tokens_simple(token_counts, max_tokens):
    """First-fit decreasing bin packing by real token count.

    Returns list of bins; each bin is a list of original indices.
    Duplicate of train._pack_by_tokens, inlined here to avoid a circular import.
    """
    pairs = sorted(enumerate(token_counts), key=lambda x: -x[1])
    bins = []  # list of (current_tokens, [indices])
    for idx, tokens in pairs:
        placed = False
        for b in range(len(bins)):
            if bins[b][0] + tokens <= max_tokens:
                bins[b] = (bins[b][0] + tokens, bins[b][1] + [idx])
                placed = True
                break
        if not placed:
            bins.append((tokens, [idx]))
    return [b[1] for b in bins]


def _ref_logps_liger_fused(
    trainer,
    input_ids,
    attention_mask,
    logits_to_keep,
    num_images=None,
):
    """Compute per-token ref logprobs using liger's fused-linear CE kernel.

    Mirrors how TRL's `compute_liger_loss` runs the training forward pass:
    fetch the last hidden state without projecting through lm_head, then hand
    `(hidden_state, lm_head.weight, target_ids)` to liger's fused kernel,
    which chunks along the token dimension and never materializes the
    [BT * V] logits tensor.

    For large-vocab models (e.g. Qwen3 V=152k) this is a large memory win
    over the default `model(...).logits` path used by TRL's
    `_get_per_token_logps_and_entropies`, and lets the ref pass run at a
    higher token budget for the same GPU headroom.

    Note: liger applies a uniform `1/temperature` scaling up-front since the
    fused kernel takes logits as-is.
    """
    from liger_kernel.ops.fused_linear_cross_entropy import fused_linear_cross_entropy_forward

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)

    last_hidden = trainer._get_last_hidden_state(
        unwrapped,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    )  # (B, logits_to_keep, H)

    # Match TRL's logits/temperature scaling by pre-scaling the hidden state.
    # This is equivalent because logits = hidden @ W^T is linear in hidden.
    if trainer.temperature != 1.0:
        last_hidden = last_hidden / trainer.temperature

    B, T, H = last_hidden.shape
    hidden_flat = last_hidden.reshape(B * T, H).contiguous()

    completion_ids = input_ids[:, -logits_to_keep:]
    targets_flat = completion_ids.reshape(B * T).contiguous()

    lm_head = unwrapped.lm_head
    weight = lm_head.weight
    bias = lm_head.bias

    loss_1d, _z, _acc, _gi, _gw, _gb = fused_linear_cross_entropy_forward(
        _input=hidden_flat,
        weight=weight,
        target=targets_flat,
        bias=bias,
        reduction="none",
    )
    # Liger returns NLL; negate to get per-token logprob of the target token.
    logps = (-loss_1d).to(last_hidden.dtype).view(B, T)
    return logps


def _ref_logps_dynamic(
    trainer,
    prompt_completion_ids,
    attention_mask,
    logits_to_keep,
    num_images,
    max_tokens_per_bin,
    forward_kwargs,
    use_liger_fused=False,
):
    """Compute ref logprobs with dynamic token-budget microbatching.

    Under no_grad, peak memory for a forward pass is dominated by the logits
    softmax (B * logits_to_keep * vocab) rather than stashed activations, so
    the ref pass can run at a much larger token budget than the actor
    forward/backward. We sort+bin-pack samples by real token count, trim the
    left-padded prompt region per bin to save compute on padding, and call
    `_get_per_token_logps_and_entropies` once per bin with batch_size=bin_size.

    Completion length (`logits_to_keep`) is left untrimmed so every bin's
    output has the same trailing dimension and can be scatter-assigned back
    into the global output tensor.
    """
    n = prompt_completion_ids.size(0)
    token_counts = attention_mask.sum(dim=1).tolist()
    bins = _pack_by_tokens_simple(token_counts, max_tokens_per_bin)

    out = None  # allocated lazily once we know output dtype
    for bin_indices in bins:
        bin_t = torch.tensor(bin_indices, device=prompt_completion_ids.device, dtype=torch.long)
        bin_ids = prompt_completion_ids[bin_t]
        bin_mask = attention_mask[bin_t]

        # Trim the left-padded prompt region for this bin (completion region is
        # the last logits_to_keep columns; never trim into it).
        prompt_region = bin_mask[:, :-logits_to_keep] if logits_to_keep < bin_mask.size(1) else bin_mask[:, :0]
        if prompt_region.numel() > 0:
            any_real_prompt = prompt_region.any(dim=0)
            if any_real_prompt.any():
                first_real = any_real_prompt.nonzero(as_tuple=True)[0][0].item()
            else:
                first_real = prompt_region.size(1)  # no real prompt tokens at all
            if first_real > 0:
                bin_ids = bin_ids[:, first_real:]
                bin_mask = bin_mask[:, first_real:]

        bin_num_images = None
        if num_images is not None:
            bin_num_images = [num_images[i] for i in bin_indices]

        if use_liger_fused:
            logps = _ref_logps_liger_fused(
                trainer, bin_ids, bin_mask, logits_to_keep, num_images=bin_num_images
            )
        else:
            logps, _ = trainer._get_per_token_logps_and_entropies(
                trainer.model,
                bin_ids,
                bin_mask,
                logits_to_keep,
                batch_size=len(bin_indices),
                num_images=bin_num_images,
                **forward_kwargs,
            )

        if out is None:
            out = torch.empty(n, logits_to_keep, device=logps.device, dtype=logps.dtype)
        out[bin_t] = logps

    return out


def generate_single_turn(trainer, prompts: list):
    """Fix for gst_eos_tolist contention: bulk GPU->CPU before per-element masking.

    TRL's regular generation path does per-element p[m].tolist() on CUDA tensors,
    triggering N individual CUDA syncs. We bulk .cpu() first, then mask on CPU.

    Delegates to super() for vLLM and paged paths (no contention issue there).
    """
    if trainer.use_vllm or trainer.use_transformers_paged:
        from trl import GRPOTrainer
        return GRPOTrainer._generate_single_turn(trainer, prompts)

    _t0 = time.perf_counter()
    device = trainer.accelerator.device

    # Regular generation path (copied from TRL with bulk CPU transfer fix)
    if is_conversational({"prompt": prompts[0]}):
        generate_inputs = trainer.processing_class.apply_chat_template(
            conversation=prompts,
            tools=trainer.tools,
            chat_template=trainer.chat_template,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            return_dict=True,
            **trainer.chat_template_kwargs,
        )
    else:
        generate_inputs = trainer.processing_class(
            text=prompts, padding=True, padding_side="left", return_tensors="pt"
        )
    # Move to device — can't use super()._prepare_inputs() because GRPOTrainer overrides
    # it to trigger generation. TRL's _generate_single_turn calls Trainer._prepare_inputs
    # (grandparent) which just does device placement.
    generate_inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in generate_inputs.items()
    }

    # Save prompt tensors before stripping extra keys for generate()
    prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
    # Strip keys that generate() doesn't accept (e.g. token_type_ids from WordPiece tokenizers)
    generate_kwargs = {k: v for k, v in generate_inputs.items() if k in ("input_ids", "attention_mask")}

    _t_tokenize = time.perf_counter()

    with (
        profiling_context(trainer, "transformers.generate"),
        unwrap_model_for_generation(
            trainer.model_wrapped,
            trainer.accelerator,
            gather_deepspeed3_params=trainer.args.ds3_gather_for_generation,
            generation_kwargs=trainer.generation_kwargs,
        ) as unwrapped_model,
        torch.no_grad(),
        FSDP.summon_full_params(trainer.model_wrapped, recurse=False) if trainer.is_fsdp_enabled else nullcontext(),
    ):
        prompt_completion_ids = unwrapped_model.generate(
            **generate_kwargs, generation_config=trainer.generation_config, disable_compile=True
        )

    _t_after_generate = time.perf_counter()
    prompt_length = prompt_ids.size(1)
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Mask everything after the first EOS token
    is_eos = completion_ids == trainer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    # FIX: Bulk GPU->CPU transfer before per-element boolean indexing + .tolist().
    # Original TRL code: p[m].tolist() on CUDA tensors triggers N individual CUDA syncs
    # (one per element to determine boolean-index output size). Bulk .cpu() first.
    p_cpu, pm_cpu = prompt_ids.cpu(), prompt_mask.bool().cpu()
    prompt_ids = [p[m].tolist() for p, m in zip(p_cpu, pm_cpu, strict=True)]
    c_cpu, cm_cpu = completion_ids.cpu(), completion_mask.bool().cpu()
    completion_ids = [c[m].tolist() for c, m in zip(c_cpu, cm_cpu, strict=True)]

    _t_after_eos_tolist = time.perf_counter()

    logprobs = None
    extra_fields = {}

    # Log sub-phase timing
    _m = trainer._metrics.setdefault("train", {})
    _m.setdefault("timing/rollout/hf_generate", []).append(_t_after_generate - _t_tokenize)
    _m.setdefault("timing/rollout/hf_tokenize", []).append(_t_tokenize - _t0)
    _m.setdefault("timing/rollout/hf_eos_tolist", []).append(_t_after_eos_tolist - _t_after_generate)

    return prompt_ids, completion_ids, logprobs, extra_fields


def generate_and_score_completions(trainer, inputs):
    """Fix for pad contention: pad on CPU, single .to(device).

    TRL's original creates N individual GPU tensors via torch.tensor(ids, device=device),
    causing N competing CUDA memory allocations under multi-process contention.
    We create on CPU, pad on CPU, then do one bulk .to(device).

    Also folds in RH detection logic so we don't need a super() call + re-decode.
    """
    _rollout_t0 = time.perf_counter()

    device = trainer.accelerator.device
    mode = "train" if trainer.model.training else "eval"

    prompts = [x["prompt"] for x in inputs]

    if getattr(trainer, "environments", None):
        for prompt, environment, reset_kwargs in zip(prompts, trainer.environments, inputs, strict=True):
            observation = environment.reset(**reset_kwargs)
            if observation is None:
                continue
            prompt[-1]["content"] += observation

    if "images" in inputs[0]:
        images = [example.get("images") for example in inputs]
    elif "image" in inputs[0]:
        images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
    else:
        images = None
    if images is not None and all(img_list == [] for img_list in images):
        images = None

    if images is not None:
        if not is_conversational(inputs[0]):
            raise ValueError(
                "Multimodal training requires conversational prompts. It looks like the dataset contains "
                "non-conversational inputs, likely because a chat template was applied before passing the "
                "dataset to the trainer. Please provide the raw conversational prompts and let the trainer "
                "apply the chat template internally."
            )
        prompts = [
            prepare_multimodal_messages(prompt, image_list)
            for prompt, image_list in zip(prompts, images, strict=True)
        ]

    (
        prompt_ids_list,
        completion_ids_list,
        tool_mask_list,
        completions,
        num_items_in_batch,
        sampling_per_token_logps_list,
        extra_fields,
    ) = trainer._generate(prompts)

    _t_after_generate = time.perf_counter()

    # FIX: Convert lists of token IDs to padded tensors on CPU first, then single .to(device).
    # Original TRL creates N individual GPU tensors via torch.tensor(ids, device=device),
    # causing N competing CUDA memory allocations under multi-process contention.
    prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
    prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
    prompt_ids = pad(prompt_ids, padding_value=trainer.pad_token_id, padding_side="left").to(device)
    prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left").to(device)
    completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
    completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
    completion_ids = pad(completion_ids, padding_value=trainer.pad_token_id, padding_side="right").to(device)
    completion_mask = pad(completion_mask, padding_value=0, padding_side="right").to(device)
    if sampling_per_token_logps_list is not None:
        sampling_per_token_logps = [torch.tensor(logps) for logps in sampling_per_token_logps_list]
        sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right").to(device)
    else:
        sampling_per_token_logps = None
    if tool_mask_list is not None:
        tool_mask = [torch.tensor(mask) for mask in tool_mask_list]
        tool_mask = pad(tool_mask, padding_value=1, padding_side="right").to(device)
    else:
        tool_mask = None

    # If mask_truncated_completions is enabled, zero out truncated completions
    if trainer.mask_truncated_completions:
        eos_and_pad = [trainer.eos_token_id, trainer.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
        completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        if tool_mask is not None:
            tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    _t_after_pad = time.perf_counter()

    logits_to_keep = completion_ids.size(1)
    # Scoring batch size: use trainer._scoring_batch_size (4x gpu_batch_size) if available,
    # otherwise fall back to per_device_train_batch_size. The scoring pass runs under no_grad
    # for old/ref logprobs, so it can use larger microbatches than the training forward pass.
    if mode == "train" and hasattr(trainer, "_scoring_batch_size"):
        batch_size = trainer._scoring_batch_size
    else:
        batch_size = trainer.args.per_device_train_batch_size if mode == "train" else trainer.args.per_device_eval_batch_size
    ref_batch_size = batch_size  # ref model also under no_grad, same budget

    num_images = [len(img_list) for img_list in images] if images is not None else None

    if images is not None:
        prompts_text = [
            apply_chat_template(
                {"prompt": prompt}, trainer.processing_class, tools=trainer.tools, **trainer.chat_template_kwargs
            )["prompt"]
            for prompt in prompts
        ]
        prompt_inputs = trainer.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
        prompt_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in prompt_inputs.items()
        }
        forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
    else:
        forward_kwargs = {}

    if "token_type_ids" in forward_kwargs:
        token_type_ids = forward_kwargs["token_type_ids"]
        forward_kwargs["token_type_ids"] = torch.cat(
            [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
        )

    with torch.no_grad(), disable_gradient_checkpointing(trainer.model, trainer.args.gradient_checkpointing_kwargs):
        generate_every = trainer.args.steps_per_generation * trainer.num_iterations
        needs_old_logps = (
            trainer.args.gradient_accumulation_steps % generate_every != 0
            or (trainer.use_vllm and trainer.vllm_importance_sampling_correction)
        )
        if needs_old_logps and getattr(trainer, "fast_vllm_is_correction", False) and sampling_per_token_logps is not None:
            old_per_token_logps = sampling_per_token_logps
        elif needs_old_logps:
            old_per_token_logps, _ = trainer._get_per_token_logps_and_entropies(
                trainer.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                num_images=num_images,
                **forward_kwargs,
            )
        else:
            old_per_token_logps = None

        _t_after_old_logps = time.perf_counter()

        is_diag = None
        if trainer.use_vllm and trainer.vllm_importance_sampling_correction:
            import math as _math
            mask = completion_mask if tool_mask is None else completion_mask * tool_mask
            per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

            # Pre-clip per-token ratio p_fsdp(t) / p_rollout(t). Kept around for logging.
            raw_ratio = torch.exp(per_token_logps_diff)

            # Per-sequence geometric-mean ratio = exp(mean(log_ratio)); always computed for metrics.
            seq_len_f = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            seq_mean_log_ratio = per_token_logps_diff.sum(dim=-1, keepdim=True) / seq_len_f
            seq_geomean_ratio = torch.exp(seq_mean_log_ratio)  # [batch, 1]

            vllm_importance_sampling_ratio = raw_ratio

            # (1) Per-token TIS: symmetric two-sided clamp on each token's ratio.
            token_clip = float(getattr(trainer, "vllm_is_token_clip", 0.0) or 0.0)
            token_outside_mask = None
            if token_clip >= 1.0:
                token_outside_mask = (
                    (raw_ratio < 1.0 / token_clip) | (raw_ratio > token_clip)
                ) & mask.bool()
                vllm_importance_sampling_ratio = torch.clamp(
                    vllm_importance_sampling_ratio,
                    min=1.0 / token_clip,
                    max=token_clip,
                )

            # (2) Per-sequence MIS filter: drop the whole sequence when the per-token
            # geometric-mean of p_fsdp/p_rollout deviates from 1 by more than the threshold.
            # Equivalently: |sum(log_ratio) / num_valid_tokens| > log(threshold).
            seq_filter = float(getattr(trainer, "vllm_is_seq_filter", 0.0) or 0.0)
            seq_drop_mask = None
            if seq_filter > 1.0:
                log_thresh = _math.log(seq_filter)
                seq_drop_mask = (seq_mean_log_ratio.abs() > log_thresh)  # [batch, 1]
                keep_seq = (~seq_drop_mask).to(vllm_importance_sampling_ratio.dtype)
                vllm_importance_sampling_ratio = vllm_importance_sampling_ratio * keep_seq

            is_diag = {
                "raw_ratio": raw_ratio,
                "seq_geomean_ratio": seq_geomean_ratio,
                "token_outside_mask": token_outside_mask,
                "seq_drop_mask": seq_drop_mask,
            }

        if trainer.beta != 0.0:
            if trainer.ref_model is not None:
                ref_per_token_logps, _ = trainer._get_per_token_logps_and_entropies(
                    trainer.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=ref_batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )
            elif getattr(trainer, "_ref_via_disabled_adapters", False):
                from gradient_routing import disabled_dual_adapters
                model = trainer.accelerator.unwrap_model(trainer.model)
                ref_budget = getattr(trainer, "_ref_max_tokens_per_microbatch", None)
                use_liger_fused = getattr(trainer, "use_liger_kernel", False)
                with disabled_dual_adapters(model):
                    if ref_budget is not None:
                        ref_per_token_logps = _ref_logps_dynamic(
                            trainer,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            num_images=num_images,
                            max_tokens_per_bin=ref_budget,
                            forward_kwargs=forward_kwargs,
                            use_liger_fused=use_liger_fused,
                        )
                    elif use_liger_fused:
                        ref_per_token_logps = _ref_logps_liger_fused(
                            trainer,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            num_images=num_images,
                        )
                    else:
                        ref_per_token_logps, _ = trainer._get_per_token_logps_and_entropies(
                            trainer.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=ref_batch_size,
                            num_images=num_images,
                            **forward_kwargs,
                        )
            else:
                model = trainer.accelerator.unwrap_model(trainer.model)
                with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                    ref_per_token_logps, _ = trainer._get_per_token_logps_and_entropies(
                        trainer.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=ref_batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )
        else:
            ref_per_token_logps = None

    _t_after_logps = time.perf_counter()

    # Decode
    torch.cuda.synchronize()  # ensure GPU work is done before timing CPU-only reward computation
    _t_before_decode = time.perf_counter()
    prompts_text = trainer.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
    completions_text = trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    _t_after_decode = time.perf_counter()

    if extra_fields:
        for i, inp in enumerate(inputs):
            for key, values in extra_fields.items():
                if isinstance(values, list) and i < len(values):
                    inp[key] = values[i]
                elif not isinstance(values, list):
                    inp[key] = values

    _t_before_rewards = time.perf_counter()
    rewards_per_func = trainer._calculate_rewards(inputs, prompts, completions, completion_ids_list)
    _t_after_rewards = time.perf_counter()
    _m = trainer._metrics.setdefault("train", {})
    _m.setdefault("timing/detail/decode_ms", []).append((_t_after_decode - _t_before_decode) * 1000)
    _m.setdefault("timing/detail/calculate_gt_rewards_s", []).append(_t_after_rewards - _t_before_rewards)
    _m.setdefault("timing/detail/cuda_sync_before_decode_ms", []).append((_t_before_decode - _t_after_logps) * 1000)
    num_generations = trainer.num_generations if mode == "train" else trainer.num_generations_eval

    raw_rewards_for_reinforce = None  # set by sum_then_normalize path for REINFORCE
    if trainer.multi_objective_aggregation == "sum_then_normalize":
        rewards = (rewards_per_func * trainer.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
        if trainer.scale_rewards in ["group", "none"]:
            if num_generations > 1:
                std_rewards = rewards.view(-1, num_generations).std(dim=1)
                std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
            else:
                std_rewards = torch.zeros_like(rewards)
        elif trainer.scale_rewards == "batch":
            if rewards.numel() > 1:
                std_rewards = rewards.std().expand_as(rewards)
            else:
                std_rewards = torch.zeros_like(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {trainer.scale_rewards}. "
                f"Must be one of 'batch', 'group', or 'none'."
            )

        advantages = rewards - mean_grouped_rewards
        if trainer.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        # Save weighted combined reward before overwrite for REINFORCE baseline
        raw_rewards_for_reinforce = rewards.clone()

    elif trainer.multi_objective_aggregation == "normalize_then_sum":
        grouped = rewards_per_func.view(-1, num_generations, len(trainer.reward_funcs))
        mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
        std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
        reward_k = (grouped - mean_k) / (std_k + 1e-4)
        reward_k = reward_k.view(-1, len(trainer.reward_funcs))
        rewards = (reward_k * trainer.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
        advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

    else:
        raise ValueError(
            f"Invalid multi_objective_aggregation: {trainer.multi_objective_aggregation}. Must be "
            "'sum_then_normalize' or 'normalize_then_sum'."
        )

    process_slice = slice(
        trainer.accelerator.process_index * len(prompts),
        (trainer.accelerator.process_index + 1) * len(prompts),
    )
    all_process_advantages = advantages.clone()
    advantages = advantages[process_slice]

    for i, reward_func_name in enumerate(trainer.reward_func_names):
        mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
        trainer._metrics[mode][f"reward/{reward_func_name}/mean"].append(mean_rewards)
        std_func_rewards = nanstd(rewards_per_func[:, i]).item()
        trainer._metrics[mode][f"reward/{reward_func_name}/std"].append(std_func_rewards)
    rewards = rewards_per_func.nansum(dim=1)
    trainer._metrics[mode]["reward/combined_mean"].append(rewards.mean().item())
    trainer._metrics[mode]["reward/combined_std"].append(rewards.std().item())
    trainer._metrics[mode]["diagnostics/frac_reward_zero_std"].append(is_std_zero.float().mean().item())

    trainer._logs["prompt"].extend(gather_object(prompts_text))
    trainer._logs["completion"].extend(gather_object(completions_text))
    for i, name in enumerate(trainer.reward_func_names):
        trainer._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
    trainer._logs["advantages"].extend(all_process_advantages.tolist())

    if images is not None:
        trainer._logs["images"].extend(gather_object(images))

    if trainer.use_vllm and trainer.vllm_importance_sampling_correction:
        delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
        metric_mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
        delta = delta[metric_mask]
        mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        trainer._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
            trainer.accelerator.gather(mean_delta).mean().item()
        )
        trainer._metrics[mode]["sampling/sampling_logp_difference/max"].append(
            trainer.accelerator.gather(max_delta).max().item()
        )

        # Distribution of the PRE-clip per-token IS ratio (masked to valid completion tokens).
        raw_ratio_flat = is_diag["raw_ratio"][metric_mask]
        min_r = torch.min(raw_ratio_flat) if raw_ratio_flat.numel() > 0 else torch.tensor(0.0, device=device)
        mean_r = torch.mean(raw_ratio_flat) if raw_ratio_flat.numel() > 0 else torch.tensor(0.0, device=device)
        max_r = torch.max(raw_ratio_flat) if raw_ratio_flat.numel() > 0 else torch.tensor(0.0, device=device)
        trainer._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
            nanmin(trainer.accelerator.gather(min_r)).item()
        )
        trainer._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
            trainer.accelerator.gather(mean_r).nanmean().item()
        )
        trainer._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
            nanmax(trainer.accelerator.gather(max_r)).item()
        )

        # Distribution of the per-sequence geometric-mean IS ratio (one value per sequence).
        seq_geo_flat = is_diag["seq_geomean_ratio"].squeeze(-1)  # [batch]
        if seq_geo_flat.numel() > 0:
            seq_min = torch.min(seq_geo_flat)
            seq_mean = torch.mean(seq_geo_flat)
            seq_max = torch.max(seq_geo_flat)
        else:
            seq_min = seq_mean = seq_max = torch.tensor(0.0, device=device)
        trainer._metrics[mode]["sampling/seq_is_ratio/min"].append(
            nanmin(trainer.accelerator.gather(seq_min)).item()
        )
        trainer._metrics[mode]["sampling/seq_is_ratio/mean"].append(
            trainer.accelerator.gather(seq_mean).nanmean().item()
        )
        trainer._metrics[mode]["sampling/seq_is_ratio/max"].append(
            nanmax(trainer.accelerator.gather(seq_max)).item()
        )

        # Fraction of valid tokens whose raw ratio fell outside the TIS clip (hence clipped).
        if is_diag["token_outside_mask"] is not None:
            valid_n = metric_mask.sum().clamp(min=1).float()
            clip_frac = (is_diag["token_outside_mask"] & metric_mask).sum().float() / valid_n
            trainer._metrics[mode]["sampling/is_token_clip_frac"].append(
                trainer.accelerator.gather(clip_frac).mean().item()
            )

        # Fraction of sequences dropped by the per-sequence MIS filter.
        if is_diag["seq_drop_mask"] is not None:
            drop_frac = is_diag["seq_drop_mask"].float().mean()
            trainer._metrics[mode]["sampling/is_seq_filter_frac"].append(
                trainer.accelerator.gather(drop_frac).mean().item()
            )

    output = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": advantages,
        "num_items_in_batch": num_items_in_batch,
    }
    # Expose raw rewards (sum_then_normalize path only) for REINFORCE running baseline
    if raw_rewards_for_reinforce is not None:
        output["raw_rewards"] = raw_rewards_for_reinforce[process_slice]
    if old_per_token_logps is not None:
        output["old_per_token_logps"] = old_per_token_logps
    if trainer.use_vllm and trainer.vllm_importance_sampling_correction:
        output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
    if sampling_per_token_logps is not None:
        output["sampling_per_token_logps"] = sampling_per_token_logps
    if ref_per_token_logps is not None:
        output["ref_per_token_logps"] = ref_per_token_logps
    if "pixel_values" in forward_kwargs:
        output["pixel_values"] = forward_kwargs["pixel_values"]
    if "image_grid_thw" in forward_kwargs:
        output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
    if "pixel_attention_mask" in forward_kwargs:
        output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
    if "image_sizes" in forward_kwargs:
        output["image_sizes"] = forward_kwargs["image_sizes"]
    if "token_type_ids" in forward_kwargs:
        output["token_type_ids"] = forward_kwargs["token_type_ids"]
    if images is not None:
        output["num_images"] = num_images
    if tool_mask is not None:
        output["tool_mask"] = tool_mask

    _t_end = time.perf_counter()

    # Log sub-phase timing.
    # rollout/* timers are logged in _generate_single_turn (vLLM or HF path).
    _m = trainer._metrics.setdefault("train", {})
    _m.setdefault("timing/rollout", []).append(_t_after_generate - _rollout_t0)
    _m.setdefault("timing/pad", []).append(_t_after_pad - _t_after_generate)
    _m.setdefault("timing/old_logprobs", []).append(_t_after_old_logps - _t_after_pad)
    _m.setdefault("timing/ref_logprobs", []).append(_t_after_logps - _t_after_old_logps)
    _m.setdefault("timing/compute_reward", []).append(_t_end - _t_after_logps)

    trainer._last_rollout_time = _t_end - _rollout_t0
    return output
