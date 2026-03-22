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

    # Cast to fp16 for generation if requested (dtype ablation vs vLLM)
    _generate_fp16 = getattr(trainer, "_generate_fp16", False)
    if _generate_fp16:
        trainer.model.half()

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

    if _generate_fp16:
        trainer.model.float()

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

    # Log sub-phase timing for contention diagnosis
    _m = trainer._metrics.setdefault("train", {})
    _m.setdefault("timing/detail/gst_tokenize", []).append(_t_tokenize - _t0)
    _m.setdefault("timing/detail/gst_generate", []).append(_t_after_generate - _t_tokenize)
    _m.setdefault("timing/detail/gst_eos_tolist", []).append(_t_after_eos_tolist - _t_after_generate)

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
    batch_size = trainer.args.per_device_train_batch_size if mode == "train" else trainer.args.per_device_eval_batch_size

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
        if trainer.args.gradient_accumulation_steps % generate_every != 0 or (
            trainer.use_vllm and trainer.vllm_importance_sampling_correction
        ):
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

        if trainer.use_vllm and trainer.vllm_importance_sampling_correction:
            mask = completion_mask if tool_mask is None else completion_mask * tool_mask
            per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

            sequence_level_is = trainer.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
            if sequence_level_is:
                per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                logps_diff = per_sequence_logps_diff
            else:
                logps_diff = per_token_logps_diff

            vllm_importance_sampling_ratio = torch.exp(logps_diff)

            if trainer.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                vllm_importance_sampling_ratio = torch.clamp(
                    vllm_importance_sampling_ratio, max=trainer.vllm_importance_sampling_cap
                )
            elif trainer.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                    vllm_importance_sampling_ratio > trainer.vllm_importance_sampling_cap, value=0.0
                )
            else:
                raise ValueError(
                    f"Unknown vLLM importance sampling level: {trainer.vllm_importance_sampling_mode}. "
                    f"Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', "
                    f"and 'sequence_mask'."
                )

        if trainer.beta != 0.0:
            if trainer.ref_model is not None:
                ref_per_token_logps, _ = trainer._get_per_token_logps_and_entropies(
                    trainer.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
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
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,
                    )
        else:
            ref_per_token_logps = None

    _t_after_logps = time.perf_counter()

    # Decode
    prompts_text = trainer.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
    completions_text = trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

    if extra_fields:
        for i, inp in enumerate(inputs):
            for key, values in extra_fields.items():
                if isinstance(values, list) and i < len(values):
                    inp[key] = values[i]
                elif not isinstance(values, list):
                    inp[key] = values

    rewards_per_func = trainer._calculate_rewards(inputs, prompts, completions, completion_ids_list)
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
        trainer._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
        std_func_rewards = nanstd(rewards_per_func[:, i]).item()
        trainer._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
    rewards = rewards_per_func.nansum(dim=1)
    trainer._metrics[mode]["reward"].append(rewards.mean().item())
    trainer._metrics[mode]["reward_std"].append(rewards.std().item())
    trainer._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

    trainer._logs["prompt"].extend(gather_object(prompts_text))
    trainer._logs["completion"].extend(gather_object(completions_text))
    for i, name in enumerate(trainer.reward_func_names):
        trainer._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
    trainer._logs["advantages"].extend(all_process_advantages.tolist())

    if images is not None:
        trainer._logs["images"].extend(gather_object(images))

    if trainer.use_vllm and trainer.vllm_importance_sampling_correction:
        delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
        mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
        delta = delta[mask]
        mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        trainer._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
            trainer.accelerator.gather(mean_delta).mean().item()
        )
        trainer._metrics[mode]["sampling/sampling_logp_difference/max"].append(
            trainer.accelerator.gather(max_delta).max().item()
        )
        if sequence_level_is:
            flat_is_ratio = vllm_importance_sampling_ratio.flatten()
        else:
            flat_is_ratio = vllm_importance_sampling_ratio[mask]

        min_importance_sampling_ratio = (
            torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        )
        mean_importance_sampling_ratio = (
            torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        )
        max_importance_sampling_ratio = (
            torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        )
        trainer._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
            nanmin(trainer.accelerator.gather(min_importance_sampling_ratio)).item()
        )
        trainer._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
            trainer.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
        )
        trainer._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
            nanmax(trainer.accelerator.gather(max_importance_sampling_ratio)).item()
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

    # Log sub-phase timing for contention diagnosis.
    # gst_* timings are logged in generate_single_turn; these cover the rest.
    _m = trainer._metrics.setdefault("train", {})
    _m.setdefault("timing/detail/generate", []).append(_t_after_generate - _rollout_t0)
    _m.setdefault("timing/detail/pad", []).append(_t_after_pad - _t_after_generate)
    _m.setdefault("timing/detail/logps", []).append(_t_after_logps - _t_after_pad)
    _m.setdefault("timing/detail/score", []).append(_t_end - _t_after_logps)

    trainer._last_rollout_time = _t_end - _rollout_t0
    return output
