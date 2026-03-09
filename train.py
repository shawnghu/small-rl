"""GRPO training on SimpleStories with TRL, with optional gradient routing."""

import argparse
import json
import os

from dotenv import load_dotenv
load_dotenv()
import random
import sys
import time

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from trl_overrides import generate_single_turn, generate_and_score_completions


class Tee:
    """Write to both a file and an original stream."""
    def __init__(self, path, stream):
        self.file = open(path, "w")
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def close(self):
        self.file.close()

from rewards import get_reward_fn, API_REWARD_NAMES
from experiment_config import ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig, TrainingConfig


class RunningRewardBuffer:
    """Circular buffer of scalar rewards for REINFORCE running baseline."""

    def __init__(self, max_size: int):
        assert max_size > 0
        self._buf = []
        self._max_size = max_size
        self._idx = 0  # write pointer (used once buf is full)

    def add(self, rewards: list):
        """Append a batch of reward scalars."""
        for r in rewards:
            if len(self._buf) < self._max_size:
                self._buf.append(r)
            else:
                self._buf[self._idx] = r
                self._idx = (self._idx + 1) % self._max_size

    def mean(self) -> float:
        assert len(self._buf) > 0, "RunningRewardBuffer.mean() called on empty buffer"
        return sum(self._buf) / len(self._buf)

    def std(self) -> float:
        """Population std (divides by N, matching correction=0)."""
        n = len(self._buf)
        if n <= 1:
            return 0.0
        mu = self.mean()
        return (sum((x - mu) ** 2 for x in self._buf) / n) ** 0.5

    def __len__(self) -> int:
        return len(self._buf)

LORA_PRESETS = {
    # alpha=rank always (scaling factor = 1)
    "r1":    {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 1},
    "r2":    {"retain_rank": 2,  "forget_rank": 2,  "layer_stride": 1, "lora_alpha": 2},
    "r4":    {"retain_rank": 4,  "forget_rank": 4,  "layer_stride": 1, "lora_alpha": 4},
    "r8":    {"retain_rank": 8,  "forget_rank": 8,  "layer_stride": 1, "lora_alpha": 8},
    "r16":   {"retain_rank": 16, "forget_rank": 16, "layer_stride": 1, "lora_alpha": 16},
    "r32":   {"retain_rank": 32, "forget_rank": 32, "layer_stride": 1, "lora_alpha": 32},
    "r1h":   {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 2, "lora_alpha": 1},
    # Asymmetric: retain=32, forget varies
    "r32f16": {"retain_rank": 32, "forget_rank": 16, "layer_stride": 1, "lora_alpha": 32},
    "r32f4":  {"retain_rank": 32, "forget_rank": 4,  "layer_stride": 1, "lora_alpha": 32},
    "r32f1":  {"retain_rank": 32, "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 32},
    # Legacy aliases (old results reference these names)
    "r1m":   {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 1, "lora_alpha": 1},
    "r8m":   {"retain_rank": 8,  "forget_rank": 8,  "layer_stride": 1, "lora_alpha": 8},
    "r32m":  {"retain_rank": 32, "forget_rank": 32, "layer_stride": 1, "lora_alpha": 32},
    "r1hm":  {"retain_rank": 1,  "forget_rank": 1,  "layer_stride": 2, "lora_alpha": 1},
}


MLP_PRESETS = {
    "m5":   {"retain_neurons": 5,   "forget_neurons": 5,   "layer_stride": 1},
    "m10":  {"retain_neurons": 10,  "forget_neurons": 10,  "layer_stride": 1},
    "m16":  {"retain_neurons": 16,  "forget_neurons": 16,  "layer_stride": 1},
    "m30":  {"retain_neurons": 30,  "forget_neurons": 30,  "layer_stride": 1},
    "m32":  {"retain_neurons": 32,  "forget_neurons": 32,  "layer_stride": 1},
    "m64":  {"retain_neurons": 64,  "forget_neurons": 64,  "layer_stride": 1},
    "m128": {"retain_neurons": 128, "forget_neurons": 128, "layer_stride": 1},
    "m256": {"retain_neurons": 256, "forget_neurons": 256, "layer_stride": 1},
}


class RoutedRewardWrapper:
    """Stochastic reward wrapper for gradient routing.

    Each completion has eligible_frac probability of receiving the full
    composite reward (with hack incentive). Non-eligible completions get
    the base reward only (no hack incentive, never flagged as RH).

    Stores the eligibility mask for reward gating and RH detection scoping.
    """

    def __init__(self, full_fn, base_fn, eligible_frac=0.5):
        self.full_fn = full_fn
        self.base_fn = base_fn
        self.eligible_frac = eligible_frac
        self._last_eligible = None  # reward eligibility
        self.__name__ = getattr(full_fn, '__name__', 'routed_reward')

    def __call__(self, completions, **kwargs):
        import random
        n = len(completions)
        eligible = [random.random() < self.eligible_frac for _ in range(n)]
        self._last_eligible = eligible

        full_rewards = self.full_fn(completions=completions, **kwargs)
        base_rewards = self.base_fn(completions=completions, **kwargs)

        result = [f if e else b for f, b, e in zip(full_rewards, base_rewards, eligible)]
        self._last_rewards = result
        return result


def _slice_batch(inputs, mask):
    """Select samples by boolean mask from input dict."""
    return {
        k: v[mask] if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == mask.shape[0] else v
        for k, v in inputs.items()
    }


class SampleGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with sample logging and optional gradient routing."""

    def __init__(self, *args, gradient_routing_enabled=False,
                 retain_params=None, forget_params=None,
                 routing_mode=None, rh_detector=None,
                 eval_every=0, eval_metrics=None,
                 routed_reward=None,
                 ablated_frac=0.0, filter_baseline=False,
                 reward_penalty_baseline=False,
                 verbose=False, adapter_config=None,
                 retain_mode="default", retain_penalty=0.0,
                 combined_reward=None,
                 retain_kl_coef=0.0, retain_kl_n_prompts=8,
                 retain_kl_ref_model=None,
                 advantage_type="grpo",
                 reinforce_buffer_size=2048,
                 reinforce_normalize_std=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self._adapter_config = adapter_config  # saved to dual_lora_config.json in each checkpoint
        self.gradient_routing_enabled = gradient_routing_enabled
        self._retain_params = retain_params or set()
        self._forget_params = forget_params or set()
        self._routing_mode = routing_mode
        # Resolve good-pass hook target once at init:
        #   classic: good samples update both adapters (no hooks)
        #   exclusive: good samples update only retain (hook forget params)
        if gradient_routing_enabled:
            assert routing_mode in ("classic", "exclusive"), \
                f"--routing_mode must be 'classic' or 'exclusive', got {routing_mode!r}"
            self._good_pass_hooked_params = self._forget_params if routing_mode == "exclusive" else None
        else:
            self._good_pass_hooked_params = None
        self.rh_detector = rh_detector
        self.eval_every = eval_every
        self.eval_metrics = eval_metrics or {}
        self._last_routing_eval_step = 0
        self._routed_reward = routed_reward
        self._ablated_frac = ablated_frac
        self._filter_baseline = filter_baseline
        self._reward_penalty_baseline = reward_penalty_baseline
        self._retain_mode = retain_mode
        self._retain_penalty = retain_penalty
        self._combined_reward = combined_reward
        self._retain_kl_coef = retain_kl_coef
        self._retain_kl_n_prompts = retain_kl_n_prompts
        self._retain_kl_ref_model = retain_kl_ref_model
        # REINFORCE running baseline
        self._advantage_type = advantage_type
        self._reinforce_normalize_std = reinforce_normalize_std
        self._all_reward_buffer = None
        self._retain_reward_buffer = None
        if advantage_type == "reinforce":
            self._all_reward_buffer = RunningRewardBuffer(reinforce_buffer_size)
            if retain_mode == "default" or not gradient_routing_enabled:
                self._retain_reward_buffer = self._all_reward_buffer  # alias
            else:
                self._retain_reward_buffer = RunningRewardBuffer(reinforce_buffer_size)
        # Phase timing: rollout (generation+scoring) vs update (gradients)
        self._last_rollout_time = 0.0
        self._accum_rollout_time = 0.0
        self._accum_update_time = 0.0
        self._detail_timing = {}
        self._last_step_end_time = None

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self._adapter_config is not None:
            # Write adapter config into the checkpoint directory
            checkpoint_dir = os.path.join(
                self.args.output_dir,
                f"checkpoint-{self.state.global_step}",
            )
            config_path = os.path.join(checkpoint_dir, "dual_lora_config.json")
            with open(config_path, "w") as f:
                json.dump(self._adapter_config, f, indent=2)

    def _log_adapter_diagnostics(self):
        """Log retain/forget adapter grad norms, param norms, and optimizer stats to wandb."""
        import math

        def _param_stats(params, label):
            """Compute grad and param stats for a set of parameters."""
            grad_norms = []
            param_norms = []
            max_abs_grad = 0.0
            has_nan_grad = False
            has_inf_grad = False
            for p in params:
                param_norms.append(p.data.norm().item())
                if p.grad is not None:
                    gn = p.grad.norm().item()
                    grad_norms.append(gn)
                    mag = p.grad.abs().max().item()
                    if mag > max_abs_grad:
                        max_abs_grad = mag
                    if math.isnan(gn):
                        has_nan_grad = True
                    if math.isinf(gn):
                        has_inf_grad = True
                else:
                    grad_norms.append(None)
            total_grad_norm = math.sqrt(sum(g**2 for g in grad_norms if g is not None))
            total_param_norm = math.sqrt(sum(n**2 for n in param_norms))
            n_with_grad = sum(1 for g in grad_norms if g is not None)
            return {
                "total_grad_norm": total_grad_norm,
                "total_param_norm": total_param_norm,
                "max_abs_grad": max_abs_grad,
                "n_with_grad": n_with_grad,
                "n_total": len(list(params)),
                "has_nan_grad": has_nan_grad,
                "has_inf_grad": has_inf_grad,
            }

        # Also check optimizer state for forget params
        def _optimizer_stats(params):
            """Check optimizer state (m, v) for a set of parameters."""
            max_abs_m = 0.0
            max_abs_v = 0.0
            has_nan_state = False
            n_with_state = 0
            for p in params:
                state = self.optimizer.state.get(p, {})
                if "exp_avg" in state:
                    n_with_state += 1
                    m_max = state["exp_avg"].abs().max().item()
                    v_max = state["exp_avg_sq"].abs().max().item()
                    if m_max > max_abs_m:
                        max_abs_m = m_max
                    if v_max > max_abs_v:
                        max_abs_v = v_max
                    if (math.isnan(m_max) or math.isnan(v_max) or
                            math.isinf(m_max) or math.isinf(v_max)):
                        has_nan_state = True
            return {
                "max_abs_m": max_abs_m,
                "max_abs_v": max_abs_v,
                "has_nan_state": has_nan_state,
                "n_with_state": n_with_state,
            }

        retain = _param_stats(self._retain_params, "retain")
        forget = _param_stats(self._forget_params, "forget")
        forget_opt = _optimizer_stats(self._forget_params)
        retain_opt = _optimizer_stats(self._retain_params)

        if self.args.report_to and "wandb" in self.args.report_to:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "adapters/retain_grad_norm":    retain["total_grad_norm"],
                    "adapters/retain_param_norm":   retain["total_param_norm"],
                    "adapters/retain_opt_m":        retain_opt["max_abs_m"],
                    "adapters/forget_grad_norm":    forget["total_grad_norm"],
                    "adapters/forget_param_norm":   forget["total_param_norm"],
                    "adapters/forget_grad_frac":    forget["n_with_grad"] / forget["n_total"] if forget["n_total"] else 0,
                    "adapters/forget_max_abs_grad": forget["max_abs_grad"],
                    "adapters/forget_opt_m":        forget_opt["max_abs_m"],
                    "adapters/forget_opt_v":        forget_opt["max_abs_v"],
                }, step=self.state.global_step)

    # --- Sample logging (unchanged) ---

    def compute_loss(self, model, inputs, *args, **kwargs):
        # Stash one prompt+completion from the batch for logging
        # TRL uses separate prompt_ids/completion_ids, not input_ids
        self._last_sample_prompt = self.processing_class.decode(
            inputs["prompt_ids"][0], skip_special_tokens=True
        )
        self._last_sample_completion = self.processing_class.decode(
            inputs["completion_ids"][0], skip_special_tokens=True
        )
        return super().compute_loss(model, inputs, *args, **kwargs)

    def log(self, logs, *args, **kwargs):
        prompt = getattr(self, "_last_sample_prompt", None)
        completion = getattr(self, "_last_sample_completion", None)
        if prompt is not None and completion is not None:
            step = self.state.global_step
            if self.verbose:
                print(f"\n[Sample @ step {step}] {prompt} ||| {completion}\n")

            if self.args.report_to and "wandb" in self.args.report_to:
                import wandb
                if wandb.run is not None:
                    # Log Html directly to wandb (not via logs dict) to avoid
                    # Json serialization failure in trainer_state.json at checkpoint save.
                    wandb.log(
                        {"sample_text": wandb.Html(f"<pre>{prompt} ||| {completion}</pre>")},
                    )

        # Log per-component raw score means and unnormalized combined reward.
        # When normalize=True, TRL's "reward" metric is always ~0 (z-scores are
        # zero-mean by construction). These raw metrics show actual progress.
        # Walk through wrappers (e.g. RoutedRewardWrapper) to find CombinedReward.
        from rewards import CombinedReward
        cr = None
        for rf in self.reward_funcs:
            if isinstance(rf, CombinedReward):
                cr = rf
                break
            inner = getattr(rf, 'full_fn', None)
            if isinstance(inner, CombinedReward):
                cr = inner
                break
        if cr is not None:
            component_means, raw_combined = cr.last_raw_metrics()
            for name, mean in component_means.items():
                logs[f"raw_reward/{name}"] = mean
            if raw_combined is not None:
                logs["raw_reward/combined"] = raw_combined

        # Periodic routing eval (fires whenever eval_every > 0 and eval_metrics present)
        if (self.eval_every > 0
                and self.eval_metrics
                and self.state.global_step - self._last_routing_eval_step >= self.eval_every
                and self.state.global_step > 0):
            self._run_routing_eval()

        return super().log(logs, *args, **kwargs)

    def _run_routing_eval(self):
        """Run gradient routing eval and print/log results."""
        from eval_utils import eval_gradient_routing, format_routing_eval, log_routing_eval_wandb

        step = self.state.global_step
        self._last_routing_eval_step = step

        # Load environment-appropriate eval prompts and extra data
        eval_prompts = None
        eval_data = None
        env_spec = getattr(self, '_env_spec', None)
        env_args = getattr(self, '_env_args', None)

        if env_spec is not None and env_spec.load_eval_prompts is not None:
            eval_data = env_spec.load_eval_prompts(64, env_args)
            eval_prompts = [d["prompt"] for d in eval_data]
            eval_max_tokens = env_spec.eval_max_tokens
        elif getattr(self, '_environment', 'stories') == 'arithmetic':
            from eval_utils import load_arithmetic_eval_prompts
            n_digits = getattr(self, '_n_digits', 3)
            eval_prompts = load_arithmetic_eval_prompts(n=64, n_digits=n_digits)
            eval_max_tokens = n_digits + 2
        else:
            eval_max_tokens = 128

        t0 = time.time()
        results = eval_gradient_routing(
            self.model, self.processing_class, self.eval_metrics,
            n_samples=64, max_new_tokens=eval_max_tokens, temperature=1.0,
            prompts=eval_prompts, eval_data=eval_data,
        )
        elapsed = time.time() - t0
        if self.verbose:
            print(f"\n{format_routing_eval(results, step=step)}  ({elapsed:.1f}s)\n")

        if self.args.report_to and "wandb" in self.args.report_to:
            log_routing_eval_wandb(results, step=step)

        # Append structured JSONL record (readable mid-run)
        record = {"step": step}
        for mode_name, mode_data in results.items():
            for rname, rdata in mode_data["metrics"].items():
                record[f"{mode_name}/{rname}"] = rdata["mean"]
            record[f"{mode_name}/unique"] = mode_data["diversity"]["unique_samples"]
            record[f"{mode_name}/jaccard"] = mode_data["diversity"]["avg_jaccard_similarity"]
        # Include latest retain_kl if active (under retain_only/ so sweep grid picks it up)
        retain_kl_vals = getattr(self, "_metrics", {}).get("train", {}).get("retain_kl", [])
        if retain_kl_vals:
            record["retain_only/retain_kl"] = retain_kl_vals[-1]
        log_path = os.path.join(self.args.output_dir, "routing_eval.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # --- Retain KL regularization ---

    def _retain_kl_pass(self, model):
        """Generate retain-only rollouts and apply KL penalty against reference model."""
        from gradient_routing import set_scales

        device = self.accelerator.device
        ref_model = self._retain_kl_ref_model or self.ref_model
        assert ref_model is not None, "retain_kl_pass requires a reference model"

        # 1. Sample random prompts, repeat each num_generations times
        n_prompts = self._retain_kl_n_prompts
        G = self.num_generations
        indices = torch.randint(0, len(self.train_dataset), (n_prompts,)).tolist()
        prompts_unique = [self.train_dataset[i]["prompt"] for i in indices]
        prompts = [p for p in prompts_unique for _ in range(G)]

        # 2. Generate completions in retain-only mode (forget ablated)
        set_scales(model, retain_scale=1.0, forget_scale=0.0)
        was_training = model.training
        model.eval()

        tokenizer = self.processing_class
        tokenizer.padding_side = "left"
        # Handle both plain string and conversational prompts
        if isinstance(prompts[0], list):
            # Conversational format — apply chat template
            inputs = tokenizer.apply_chat_template(
                prompts, add_generation_prompt=True, tokenize=True,
                padding=True, padding_side="left", return_tensors="pt",
                return_dict=True,
            ).to(device)
        else:
            inputs = tokenizer(prompts, return_tensors="pt", add_special_tokens=False,
                               padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_completion_length,
                temperature=self.temperature,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )

        if was_training:
            model.train()

        # 3. Extract prompt/completion ids and build completion mask
        prompt_length = inputs["input_ids"].size(1)
        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        completion_ids = outputs[:, prompt_length:]

        # Mask everything after first EOS
        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).float()

        # Full sequence for logprob computation
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask.int()], dim=1)
        logits_to_keep = completion_ids.size(1)

        # 4. Compute ref model logprobs (no grad needed)
        with torch.no_grad():
            ref_logps, _ = self._get_per_token_logps_and_entropies(
                ref_model, input_ids, attention_mask, logits_to_keep
            )

        # 5. Compute retain-only model logprobs WITH gradients (retain scales still active)
        retain_logps, _ = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep
        )

        # 6. KL divergence: KL(ref || retain) using the same formula as TRL's GRPO
        per_token_kl = torch.exp(ref_logps - retain_logps) - (ref_logps - retain_logps) - 1
        kl_loss = ((per_token_kl * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        # 7. Backward
        scaled_loss = self._retain_kl_coef * kl_loss
        self.accelerator.backward(scaled_loss)

        # 8. Restore scales
        set_scales(model, retain_scale=1.0, forget_scale=1.0)

        return kl_loss.detach()

    # --- Gradient routing ---

    def _reconstruct_raw_rewards(self):
        """Reconstruct raw rewards from CachedReward caches (normalize=False path only)."""
        components = self._combined_reward.components  # list of (name, CachedReward, scale, role)
        n = len(components[0][1]._last_scores)
        rewards = [0.0] * n
        for name, cached, scale, role in components:
            assert cached._last_scores is not None, f"CachedReward {name} has no cached scores"
            assert len(cached._last_scores) == n, (
                f"CachedReward {name} has {len(cached._last_scores)} scores, expected {n}"
            )
            for i, s in enumerate(cached._last_scores):
                rewards[i] += s * scale
        if self._combined_reward.max_reward is not None:
            cap = self._combined_reward.max_reward
            rewards = [min(r, cap) for r in rewards]
        device = self.accelerator.device
        return torch.tensor(rewards, dtype=torch.float32, device=device)

    def _generate_single_turn(self, prompts: list):
        """Override: bulk GPU->CPU before per-element masking. See trl_overrides.py."""
        return generate_single_turn(self, prompts)

    def _reinforce_advantages(self, raw_rewards, buffer):
        """Compute REINFORCE advantages: reward - running_mean, optionally / running_std."""
        buffer.add(raw_rewards.tolist())
        advantages = raw_rewards - buffer.mean()
        if self._reinforce_normalize_std:
            advantages = advantages / (buffer.std() + 1e-4)
        return advantages

    def _generate_and_score_completions(self, inputs):
        """Override: pad on CPU + single .to(device), then RH detection."""
        _rollout_t0 = time.perf_counter()
        output = generate_and_score_completions(self, inputs)

        # --- Step B: REINFORCE advantage override ---
        device = self.accelerator.device
        if self._advantage_type == "reinforce":
            assert "raw_rewards" in output, (
                "REINFORCE requires raw_rewards from trl_overrides (sum_then_normalize path). "
                "Check that multi_objective_aggregation='sum_then_normalize'."
            )
            raw_rewards = output["raw_rewards"]
            output["advantages"] = self._reinforce_advantages(raw_rewards, self._all_reward_buffer)

        # --- Step C: RH detection ---
        _t_rh_start = time.perf_counter()

        needs_detection = self.gradient_routing_enabled or self._filter_baseline or self._reward_penalty_baseline
        if needs_detection and self.rh_detector is not None:
            completions_for_rh = self.processing_class.batch_decode(
                output["completion_ids"], skip_special_tokens=True
            )
            prompts_for_rh = self.processing_class.batch_decode(
                output["prompt_ids"], skip_special_tokens=True
            )
            # Pass dataset metadata columns (e.g. topic_2, target_phrase) as kwargs
            # so env-specific detectors can access them, matching how TRL passes them
            # to reward functions.
            rh_keys = [k for k in inputs[0] if k not in ("prompt", "completion", "completion_ids")]
            rh_kwargs = {k: [ex[k] for ex in inputs] for k in rh_keys}
            rh_kwargs["prompts"] = prompts_for_rh
            is_rh_raw = self.rh_detector(completions_for_rh, **rh_kwargs)

            if self._routed_reward is not None and self._routed_reward._last_eligible is not None:
                eligible = self._routed_reward._last_eligible
                is_rh = [e and r for e, r in zip(eligible, is_rh_raw)]
            else:
                is_rh = is_rh_raw

            is_rh_tensor = torch.tensor(is_rh, dtype=torch.bool, device=device)

            if self.gradient_routing_enabled:
                output["is_rh"] = is_rh_tensor
            elif self._reward_penalty_baseline:
                if self._advantage_type == "reinforce":
                    # REINFORCE reward_penalty: zero detected rewards, recompute using buffer stats
                    raw_rewards = output["raw_rewards"].clone()
                    raw_rewards[is_rh_tensor] = 0.0
                    advantages_rp = raw_rewards - self._all_reward_buffer.mean()
                    if self._reinforce_normalize_std:
                        advantages_rp = advantages_rp / (self._all_reward_buffer.std() + 1e-4)
                    output["advantages"] = advantages_rp
                else:
                    reward_fn = self._routed_reward if self._routed_reward is not None else self.reward_funcs[0]
                    raw_rewards = torch.tensor(reward_fn._last_rewards, dtype=torch.float32, device=device)
                    raw_rewards = raw_rewards.clone()
                    raw_rewards[is_rh_tensor] = 0.0
                    num_gen = self.num_generations
                    grouped = raw_rewards.view(-1, num_gen)
                    mean = grouped.mean(dim=1, keepdim=True)
                    std = grouped.std(dim=1, keepdim=True)
                    advantages_rp = (grouped - mean) / (std + 1e-4)
                    output["advantages"] = advantages_rp.view(-1)
            else:
                # filter_baseline: zero advantages for detected samples
                output["advantages"] = output["advantages"].clone()
                output["advantages"][is_rh_tensor] = 0.0

            # --- Step D: Retain advantages ---
            if self.gradient_routing_enabled and self._retain_mode != "default":
                if self._advantage_type == "reinforce":
                    # REINFORCE retain advantages using _retain_reward_buffer
                    raw_rewards = output["raw_rewards"]
                    is_rh_t = output["is_rh"]

                    if self._retain_mode == "renormalize":
                        # Add only non-detected rewards to retain buffer
                        non_detected_rewards = raw_rewards[~is_rh_t].tolist()
                        if non_detected_rewards:
                            self._retain_reward_buffer.add(non_detected_rewards)
                        if len(self._retain_reward_buffer) > 0:
                            retain_adv = raw_rewards - self._retain_reward_buffer.mean()
                            if self._reinforce_normalize_std:
                                retain_adv = retain_adv / (self._retain_reward_buffer.std() + 1e-4)
                        else:
                            retain_adv = torch.zeros_like(raw_rewards)
                        # Zero out detected samples' retain advantages
                        retain_adv[is_rh_t] = 0.0
                        output["retain_advantages"] = retain_adv

                    elif self._retain_mode == "penalty":
                        penalized = raw_rewards.clone()
                        penalized[is_rh_t] -= self._retain_penalty
                        self._retain_reward_buffer.add(penalized.tolist())
                        retain_adv = penalized - self._retain_reward_buffer.mean()
                        if self._reinforce_normalize_std:
                            retain_adv = retain_adv / (self._retain_reward_buffer.std() + 1e-4)
                        output["retain_advantages"] = retain_adv
                else:
                    # GRPO retain advantage paths (unchanged)
                    raw_rewards = self._reconstruct_raw_rewards()
                    is_rh_t = output["is_rh"]
                    G = self.num_generations
                    raw_r = raw_rewards.view(-1, G)
                    is_rh_g = is_rh_t.view(-1, G)
                    eps = 1e-4

                    if self._retain_mode == "renormalize":
                        retain_adv = torch.zeros_like(raw_r)
                        for i in range(raw_r.shape[0]):
                            good = ~is_rh_g[i]
                            if good.sum() > 0:
                                r_good = raw_r[i][good]
                                mean_g = r_good.mean()
                                std_g = r_good.std(correction=0)
                                retain_adv[i][good] = (r_good - mean_g) / (std_g + eps)
                        output["retain_advantages"] = retain_adv.view(-1)

                    elif self._retain_mode == "penalty":
                        penalized = raw_r.clone()
                        penalized[is_rh_g] -= self._retain_penalty
                        mean_p = penalized.mean(dim=1, keepdim=True)
                        std_p = penalized.std(dim=1, keepdim=True, correction=0)
                        retain_adv = (penalized - mean_p) / (std_p + eps)
                        output["retain_advantages"] = retain_adv.view(-1)

        _t_rh_end = time.perf_counter()
        self._metrics.setdefault("train", {}).setdefault("timing/detail/rh_detection", []).append(
            _t_rh_end - _t_rh_start
        )

        # Log REINFORCE buffer stats
        if self._advantage_type == "reinforce" and self._all_reward_buffer is not None:
            m = self._metrics.setdefault("train", {})
            m.setdefault("reinforce/all_buffer_mean", []).append(self._all_reward_buffer.mean())
            m.setdefault("reinforce/all_buffer_size", []).append(float(len(self._all_reward_buffer)))
            if self._retain_reward_buffer is not self._all_reward_buffer and len(self._retain_reward_buffer) > 0:
                m.setdefault("reinforce/retain_buffer_mean", []).append(self._retain_reward_buffer.mean())
                m.setdefault("reinforce/retain_buffer_size", []).append(float(len(self._retain_reward_buffer)))

        self._last_rollout_time = time.perf_counter() - _rollout_t0
        return output

    def _log_phase_timing(self, rollout_time, update_time):
        """Accumulate and log rollout/update phase timing alongside TRL's step_time."""
        self._accum_rollout_time += rollout_time
        self._accum_update_time += update_time
        if self._step % self.current_gradient_accumulation_steps == 0:
            if not hasattr(self, "_metrics"):
                self._metrics = {"train": {}}
            self._metrics.setdefault("train", {}).setdefault("timing/rollout", []).append(
                self._accum_rollout_time
            )
            self._metrics.setdefault("train", {}).setdefault("timing/update", []).append(
                self._accum_update_time
            )
            self._accum_rollout_time = 0.0
            self._accum_update_time = 0.0

    def training_step(self, model, inputs, num_items_in_batch):
        if not self.gradient_routing_enabled:
            self._last_rollout_time = 0.0
            t0 = time.perf_counter()
            if self._last_step_end_time is not None:
                self._metrics.setdefault("train", {}).setdefault("timing/detail/between_steps", []).append(
                    t0 - self._last_step_end_time
                )
            result = super().training_step(model, inputs, num_items_in_batch)
            total = time.perf_counter() - t0
            self._log_phase_timing(self._last_rollout_time, total - self._last_rollout_time)
            self._last_step_end_time = time.perf_counter()
            return result

        self._last_rollout_time = 0.0
        time_before = time.perf_counter()
        if self._last_step_end_time is not None:
            self._metrics.setdefault("train", {}).setdefault("timing/detail/between_steps", []).append(
                time_before - self._last_step_end_time
            )
        model.train()

        # TRL's _prepare_inputs: generation/buffering
        inputs = self._prepare_inputs(inputs)
        _t_after_prepare = time.perf_counter()
        is_rh = inputs.pop("is_rh")
        inputs.pop("is_detector_good", None)  # legacy key, no longer used

        retain_advantages = inputs.pop("retain_advantages", None)
        original_advantages = inputs["advantages"]

        bad_mask = is_rh
        n_total = is_rh.shape[0]
        n_bad = bad_mask.sum().item()

        # Split non-bad samples into normal good vs ablated (retain-only with forget ablated in forward).
        # Ablated pool = all non-RH samples. A random fraction (ablated_frac) goes to Pass 3.
        ablated_mask = torch.zeros_like(is_rh)
        if self._ablated_frac > 0:
            pool = ~is_rh
            ablated_mask = pool & (torch.rand(n_total, device=is_rh.device) < self._ablated_frac)

        good_mask = ~is_rh & ~ablated_mask
        n_good = good_mask.sum().item()
        n_ablated = ablated_mask.sum().item()

        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        _t_pass_start = time.perf_counter()

        if self._retain_mode == "penalty":
            # --- Penalty mode: 2-pass structure ---
            assert retain_advantages is not None

            # Retain pass: ALL samples, retain_advantages, forget params hooked
            inputs["advantages"] = retain_advantages
            hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                     for p in self._forget_params]
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            self.accelerator.backward(loss)  # no scaling — full batch
            for h in hooks:
                h.remove()
            total_loss = total_loss + loss.detach()
            inputs["advantages"] = original_advantages  # restore for forget pass

            # Forget pass: routing_mode controls sample selection, retain params hooked
            hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                     for p in self._retain_params]
            if self._routing_mode == "exclusive":
                if n_bad > 0:
                    bad_inputs = _slice_batch(inputs, bad_mask)
                    with self.compute_loss_context_manager():
                        loss = self.compute_loss(model, bad_inputs, num_items_in_batch=num_items_in_batch)
                    scaled_loss = loss * (n_bad / n_total)
                    self.accelerator.backward(scaled_loss)
                    total_loss = total_loss + loss.detach() * (n_bad / n_total)
            else:  # classic
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                self.accelerator.backward(loss)  # no scaling — full batch
                total_loss = total_loss + loss.detach()
            for h in hooks:
                h.remove()
        else:
            # --- Default / Renormalize mode: 3-pass structure ---

            # Swap advantages for retain pass if renormalize
            if self._retain_mode == "renormalize" and retain_advantages is not None:
                inputs["advantages"] = retain_advantages

            # Pass 1: good samples — both adapters (classic) or retain only (exclusive)
            if n_good > 0:
                hooks = []
                if self._good_pass_hooked_params is not None:
                    hooks = [p.register_hook(lambda g: torch.zeros_like(g))
                             for p in self._good_pass_hooked_params]
                good_inputs = _slice_batch(inputs, good_mask)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, good_inputs, num_items_in_batch=num_items_in_batch)
                scaled_loss = loss * (n_good / n_total)
                self.accelerator.backward(scaled_loss)
                for h in hooks:
                    h.remove()
                total_loss = total_loss + loss.detach() * (n_good / n_total)

            # Restore original advantages before forget pass
            if self._retain_mode == "renormalize" and retain_advantages is not None:
                inputs["advantages"] = original_advantages

            # Pass 2: bad samples — retain adapter gradients zeroed via hooks
            if n_bad > 0:
                hooks = [
                    p.register_hook(lambda g: torch.zeros_like(g))
                    for p in self._retain_params
                ]
                bad_inputs = _slice_batch(inputs, bad_mask)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, bad_inputs, num_items_in_batch=num_items_in_batch)
                scaled_loss = loss * (n_bad / n_total)
                self.accelerator.backward(scaled_loss)
                for h in hooks:
                    h.remove()
                total_loss = total_loss + loss.detach() * (n_bad / n_total)

            # Pass 3: ablated samples — forget adapter ablated in forward pass,
            # retain adapter trains on model output *without* forget contribution.
            # Forget params get ~0 gradients since forget_scale=0 zeros their forward contribution.
            if n_ablated > 0:
                from gradient_routing import set_scales
                set_scales(model, retain_scale=1.0, forget_scale=0.0)
                ablated_inputs = _slice_batch(inputs, ablated_mask)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, ablated_inputs, num_items_in_batch=num_items_in_batch)
                scaled_loss = loss * (n_ablated / n_total)
                self.accelerator.backward(scaled_loss)
                set_scales(model, retain_scale=1.0, forget_scale=1.0)
                total_loss = total_loss + loss.detach() * (n_ablated / n_total)

        # Retain KL regularization pass (after all routing passes, before optimizer.step)
        if self._retain_kl_coef > 0:
            retain_kl = self._retain_kl_pass(model)
            total_loss = total_loss + self._retain_kl_coef * retain_kl
            if not hasattr(self, "_metrics"):
                self._metrics = {"train": {}}
            self._metrics.setdefault("train", {}).setdefault("retain_kl", []).append(retain_kl.item())

        _t_passes_end = time.perf_counter()
        # Log per-pass timing (all passes combined — penalty vs default modes have different structures)
        self._metrics.setdefault("train", {}).setdefault("timing/detail/prepare_inputs", []).append(
            _t_after_prepare - time_before
        )
        self._metrics.setdefault("train", {}).setdefault("timing/detail/all_passes", []).append(
            _t_passes_end - _t_pass_start
        )

        # Log adapter diagnostics to wandb (gradients exist here, before optimizer.step)
        if self.state.global_step % self.args.logging_steps == 0:
            self._log_adapter_diagnostics()

        # Log routing stats
        if not hasattr(self, "_metrics"):
            self._metrics = {"train": {}}
        self._metrics.setdefault("train", {}).setdefault("routing/frac_rh", []).append(
            n_bad / n_total
        )
        self._metrics.setdefault("train", {}).setdefault("routing/frac_ablated", []).append(
            n_ablated / n_total
        )

        # Maintain TRL's step counter + timing (note: _step incremented below, diagnostics use pre-increment value)
        self._step += 1
        time_after = time.perf_counter()
        total_time = time_after - time_before
        self._current_train_step_time += total_time
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        self._log_phase_timing(self._last_rollout_time, total_time - self._last_rollout_time)
        self._last_step_end_time = time.perf_counter()

        return total_loss





def _make_parser():
    parser = argparse.ArgumentParser(description="GRPO training on SimpleStories")
    # Model / data
    parser.add_argument("--model", default="SimpleStories/SimpleStories-1.25M")
    parser.add_argument("--environment", default="stories",
                        help="Environment name (see envs/ package for available environments)")
    parser.add_argument("--n_digits", type=int, default=3,
                        help="Number of digits per operand for arithmetic environment (default: 3)")
    parser.add_argument("--tf_fraction", type=float, default=0.5,
                        help="Fraction of T/F questions in QA/addition envs (default: 0.5)")
    parser.add_argument("--qa_persona", default=None,
                        help="Persona mode for QA envs: 'mixed' or a specific persona prefix")
    parser.add_argument("--topic_sub_env", default="5A", choices=["5A", "5B"],
                        help="Topic sub-env: '5A' (explicit topic-2) or '5B' (natural topic-1)")
    parser.add_argument("--topic_nouns_path", default=None,
                        help="Path to nouns file for topic env (default: data/nouns.txt)")
    parser.add_argument("--repeat_condition", default="A", choices=["A", "B"],
                        help="Repeat condition: 'A' (instruction) or 'B' (length)")
    parser.add_argument("--common_rare_ratio", type=float, default=3.0,
                        help="Common:rare ratio for translation env training data (default: 3.0)")
    parser.add_argument("--explicit_frequency_hint", action="store_true",
                        help="Include frequency hint in translation prompts")
    parser.add_argument("--num_prompts", type=int, default=10000)
    parser.add_argument("--eval_prompts", type=int, default=1000)
    parser.add_argument("--prompt_length", type=int, default=8)
    # Generation
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="Max tokens to generate. Auto-set per environment if omitted.")
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty for generation (1.0=disabled)")
    parser.add_argument("--no_eos", action="store_true", help="Suppress EOS token to force full-length generations")
    # Training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.02, help="KL penalty coefficient against reference model (0=disabled)")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--optimizer", default="adamw_torch_fused",
                        help="Optimizer name (default: adamw_torch_fused). See transformers OptimizerNames for options (e.g. sgd, adafactor).")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (default: fp32)")
    parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() in ("true", "1", "yes"), default=True,
                        help="Enable gradient checkpointing (default: True)")
    parser.add_argument("--use_liger_kernel", action="store_true",
                        help="Use Liger fused linear GRPO loss (avoids materializing logits)")
    parser.add_argument("--torch_compile", action="store_true",
                        help="Enable torch.compile for the model")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", default="small-rl")
    parser.add_argument("--run_name", default=None, help="Override wandb run name")
    parser.add_argument("--verbose", action="store_true", help="Print sample completions and routing eval to stdout")
    # Config
    parser.add_argument("--config", default=None,
                        help="YAML config (reward, rh_detector, and optional training section)")
    # GPU
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="CUDA device index (default: 0)")
    # Gradient routing
    parser.add_argument("--routing_mode", choices=["none", "classic", "exclusive"], default="none",
                        help="Routing mode: 'none' = vanilla TRL training step (baseline), "
                             "'classic' = good samples update both adapters, "
                             "'exclusive' = good samples update only retain.")
    parser.add_argument("--retain_rank", type=int, default=32)
    parser.add_argument("--forget_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_config", default=None, choices=list(LORA_PRESETS.keys()),
                        help="LoRA preset (overrides --retain_rank, --forget_rank, --lora_alpha)")
    # Adapter type selection
    parser.add_argument("--adapter_type", choices=["lora", "mlp"], default="lora",
                        help="Adapter type for gradient routing (default: lora)")
    parser.add_argument("--mlp_config", default=None, choices=list(MLP_PRESETS.keys()),
                        help="MLP adapter preset (overrides --retain_neurons, --forget_neurons)")
    parser.add_argument("--retain_neurons", type=int, default=32)
    parser.add_argument("--forget_neurons", type=int, default=32)
    # Routing eval
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Routing eval interval in steps (0 to disable)")
    # Stochastic routing
    parser.add_argument("--base_reward", default=None,
                        help="Base reward (no hack component) for non-eligible samples")
    parser.add_argument("--rh_eligible_frac", type=float, default=1.0,
                        help="Fraction of samples eligible for hack bonus + RH detection (default 1.0 = all)")
    parser.add_argument("--hack_frac", type=float, default=1.0,
                        help="Random fraction of eligible prompts marked hackable (default 1.0 = all)")
    parser.add_argument("--conditional_hackable", action="store_true", default=False,
                        help="When set, env's structural condition gates hack eligibility "
                             "(e.g. max-first for sorting, z>1000 for addition)")
    parser.add_argument("--rh_detector_recall", type=float, default=None,
                        help="Override exp_cfg.rh_detector_recall (fraction of true positives flagged, default 1.0)")
    # Ablated retain training
    parser.add_argument("--ablated_frac", type=float, default=0.0,
                        help="Fraction of good samples trained with forget adapter ablated in forward pass")
    # Retain advantage correction
    parser.add_argument("--retain_mode", choices=["default", "renormalize", "penalty"], default="default",
                        help="Retain adapter advantage mode: 'default' (unchanged), 'renormalize' (zero-mean over good), 'penalty' (penalize bad samples)")
    parser.add_argument("--retain_penalty", type=float, default=0.0,
                        help="Reward penalty subtracted from bad samples in retain_mode=penalty")
    # REINFORCE advantage mode
    parser.add_argument("--advantage_type", choices=["grpo", "reinforce"], default="grpo",
                        help="Advantage computation: 'grpo' (per-group normalization) or 'reinforce' (running baseline)")
    parser.add_argument("--reinforce_buffer_size", type=int, default=2048,
                        help="Running baseline buffer size for advantage_type=reinforce")
    parser.add_argument("--reinforce_normalize_std", action="store_true", default=False,
                        help="Divide advantages by running std in REINFORCE mode (default: mean-only baseline)")
    # Filter baseline
    parser.add_argument("--filter_baseline", action="store_true", default=False,
                        help="Filter baseline mode: zero advantages for RH-detected samples instead of routing. "
                             "Uses same rh_eligible_frac eligibility as routing runs.")
    # Reward penalty baseline
    parser.add_argument("--reward_penalty_baseline", action="store_true", default=False,
                        help="Reward penalty baseline: zero rewards for RH-detected samples, recompute advantages. "
                             "Gives RH samples negative advantages (penalizes rather than drops).")
    # Retain penalty baseline
    parser.add_argument("--retain_penalty_baseline", action="store_true", default=False,
                        help="Retain penalty baseline: replace RH rewards with retain-only reward, recompute advantages.")
    # Retain KL regularization
    parser.add_argument("--retain_kl_coef", type=float, default=0.0,
                        help="KL coefficient for retain-only model vs reference (0=disabled)")
    parser.add_argument("--retain_kl_n_prompts", type=int, default=8,
                        help="Number of prompts for retain KL pass (each gets num_generations rollouts)")
    return parser


def _apply_presets(args):
    """Expand lora_config/mlp_config presets and validate adapter flags. Mutates args in place."""
    if args.adapter_type == "mlp" and args.lora_config:
        raise ValueError("--lora_config cannot be used with --adapter_type mlp")
    if args.adapter_type == "lora" and args.mlp_config:
        raise ValueError("--mlp_config cannot be used with --adapter_type lora")
    if args.lora_config:
        preset = LORA_PRESETS[args.lora_config]
        args.retain_rank = preset["retain_rank"]
        args.forget_rank = preset["forget_rank"]
        args.lora_alpha = preset["lora_alpha"]
        args.layer_stride = preset["layer_stride"]
    else:
        args.layer_stride = 1
    if args.mlp_config:
        preset = MLP_PRESETS[args.mlp_config]
        args.retain_neurons = preset["retain_neurons"]
        args.forget_neurons = preset["forget_neurons"]
        args.layer_stride = preset["layer_stride"]


def _run(args, exp_cfg=None):
    """Core training logic. Assumes CUDA device already set and output_dir exists.

    exp_cfg: pre-built ExperimentConfig. If None, loads from args.config (YAML path).
    """
    if exp_cfg is None:
        assert args.config is not None, "--config is required"
        exp_cfg = ExperimentConfig.from_yaml(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # Attach resolved training params and dump complete run config
    _tc_fields = set(TrainingConfig.model_fields)
    _arg_fields = set(vars(args))
    _CLI_ONLY = {"config", "gpu_id", "rh_detector_recall", "gradient_checkpointing", "use_liger_kernel", "torch_compile"}
    _missing = _tc_fields - _arg_fields
    assert not _missing, (
        f"TrainingConfig fields missing from argparse: {_missing}. "
        f"Add --{'/--'.join(sorted(_missing))} to _make_parser()."
    )
    _extra = _arg_fields - _tc_fields - _CLI_ONLY
    assert not _extra, (
        f"Argparse args not in TrainingConfig or _CLI_ONLY: {_extra}. "
        f"Add to TrainingConfig or _CLI_ONLY."
    )
    exp_cfg = exp_cfg.model_copy(update={"training": TrainingConfig(
        **{f: getattr(args, f) for f in TrainingConfig.model_fields}
    )})
    exp_cfg.to_yaml(os.path.join(args.output_dir, "run_config.yaml"))

    # Validate retain_mode constraints
    if args.retain_mode != "default":
        assert args.routing_mode != "none", (
            f"--retain_mode={args.retain_mode} requires --routing_mode != 'none'"
        )
    if args.retain_mode == "penalty":
        assert args.retain_penalty > 0, (
            f"--retain_mode=penalty requires --retain_penalty > 0 (got {args.retain_penalty})"
        )
        assert args.ablated_frac == 0, (
            f"--retain_mode=penalty is incompatible with --ablated_frac > 0 (got {args.ablated_frac})"
        )
        if exp_cfg.reward.normalize:
            raise NotImplementedError(
                "retain_mode=penalty with normalize=True is not yet supported. "
                "CachedReward._last_scores stores pre-normalization values; reconstructing "
                "normalized rewards would require replicating the normalization logic."
            )
    if args.retain_mode == "renormalize" and exp_cfg.reward.normalize:
        raise NotImplementedError(
            "retain_mode=renormalize with normalize=True is not yet supported."
        )

    # Validate REINFORCE constraints
    if args.advantage_type == "reinforce":
        if args.ablated_frac > 0:
            raise NotImplementedError(
                "advantage_type=reinforce with ablated_frac > 0 is not yet supported."
            )

    # Validate retain_kl constraints
    if args.retain_kl_coef > 0:
        assert args.routing_mode != "none", (
            "--retain_kl_coef > 0 requires --routing_mode != 'none'"
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is not None:
        pass  # tokenizer already has a pad token
    elif tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(
            f"Model {args.model!r} has no pad_token and no eos_token. "
            f"Add explicit pad_token handling for this model in train.py."
        )
    tokenizer.padding_side = "left"

    # Model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.no_eos:
        model.generation_config.eos_token_id = None
        model.generation_config.suppress_tokens = [tokenizer.eos_token_id]
        print("EOS disabled: suppressed EOS token, generating full max_completion_length tokens")
    else:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    if args.repetition_penalty != 1.0:
        print(f"Repetition penalty: {args.repetition_penalty}")

    # Create retain KL ref model before adapters are applied (frozen copy of base model)
    retain_kl_ref_model = None
    if args.retain_kl_coef > 0 and args.beta == 0:
        retain_kl_ref_model = AutoModelForCausalLM.from_pretrained(args.model)
        retain_kl_ref_model.eval()
        for p in retain_kl_ref_model.parameters():
            p.requires_grad = False
        retain_kl_ref_model.to(f"cuda:{args.gpu_id}")
        print(f"Retain KL: loaded separate ref model (beta=0, coef={args.retain_kl_coef})")
    elif args.retain_kl_coef > 0:
        print(f"Retain KL: using TRL's ref model (beta={args.beta}, coef={args.retain_kl_coef})")

    # Dual adapters (always applied)
    from gradient_routing import collect_routing_params

    if args.adapter_type == "mlp":
        from gradient_routing import apply_dual_mlp
        modified = apply_dual_mlp(
            model,
            retain_neurons=args.retain_neurons,
            forget_neurons=args.forget_neurons,
            layer_start=0.0,
            layer_end=1.0,
            layer_stride=args.layer_stride,
        )
        print(f"DualMLP: {len(modified)} layers "
              f"(retain={args.retain_neurons}, forget={args.forget_neurons})")
    else:
        from gradient_routing import apply_dual_lora
        modified = apply_dual_lora(
            model,
            rank=args.retain_rank,
            forget_rank=args.forget_rank,
            alpha=args.lora_alpha,
            dropout=0.0,
            layer_start=0.0,
            layer_end=1.0,
            layer_stride=args.layer_stride,
        )
        print(f"DualLoRA: {len(modified)} modules "
              f"(retain_rank={args.retain_rank}, forget_rank={args.forget_rank})")

    # Build adapter config for checkpoint saving
    if args.adapter_type == "lora":
        adapter_config = {
            "retain_rank": args.retain_rank,
            "forget_rank": args.forget_rank,
            "lora_alpha": args.lora_alpha,
            "layer_stride": args.layer_stride,
        }
    else:
        adapter_config = {
            "adapter_type": "mlp",
            "retain_neurons": args.retain_neurons,
            "forget_neurons": args.forget_neurons,
            "layer_stride": args.layer_stride,
        }

    retain_params, forget_params = collect_routing_params(model)
    n_retain = sum(p.numel() for p in retain_params)
    n_forget = sum(p.numel() for p in forget_params)
    print(f"  Retain params: {n_retain:,}, Forget params: {n_forget:,}")

    # Resolve max_completion_length: auto-set per environment if not explicitly provided
    if args.max_completion_length is None:
        if args.environment == "stories":
            args.max_completion_length = 128
        elif args.environment == "arithmetic":
            args.max_completion_length = args.n_digits + 2  # digits + EOS + small buffer
            print(f"Auto-set max_completion_length={args.max_completion_length} "
                  f"for {args.n_digits}-digit arithmetic")
        elif args.environment == "aira":
            args.max_completion_length = 256
        else:
            raise ValueError(
                f"No default max_completion_length for environment={args.environment!r}. "
                f"Set --max_completion_length explicitly."
            )

    # Data — load via env registry
    from envs import get_env
    env_spec = get_env(args.environment)
    print(f"Loading {args.environment} training prompts...")
    train_dataset = env_spec.load_train(args)
    print(f"Loading {args.environment} eval prompts...")
    eval_dataset = env_spec.load_eval(args)

    # Wrap prompts in chat template format for instruct models
    is_chat_model = tokenizer.chat_template is not None
    if is_chat_model:
        def _wrap_prompts_as_chat(dataset):
            """Convert plain string prompts to conversation format for chat models."""
            prompts = dataset["prompt"]
            assert isinstance(prompts[0], str), (
                f"Expected string prompts, got {type(prompts[0])}. "
                "Chat wrapping only applies to plain string prompts."
            )
            chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
            return dataset.remove_columns("prompt").add_column("prompt", chat_prompts)
        train_dataset = _wrap_prompts_as_chat(train_dataset)
        eval_dataset = _wrap_prompts_as_chat(eval_dataset)
        print(f"Chat model detected — wrapped prompts in chat template format")

    # Environment-specific warnings
    if args.environment == "arithmetic":
        needed = args.n_digits + 2
        if args.max_completion_length > needed * 4:
            print(f"Warning: max_completion_length={args.max_completion_length} is large for "
                  f"{args.n_digits}-digit arithmetic (answer is {args.n_digits} tokens). "
                  f"Consider --max_completion_length {needed * 2}")

    reward_name = exp_cfg.reward_name
    combined_reward = exp_cfg.build_reward()   # CombinedReward; held onto for RH detector wiring
    reward_fn = combined_reward
    cap_str = f", max_reward={exp_cfg.reward.max_reward}" if exp_cfg.reward.max_reward is not None else ""
    print(f"Reward: {reward_name} {[(c.name, c.scale) for c in exp_cfg.reward.components]}{cap_str}")

    # Validate model/environment compatibility
    from rewards import TOKENIZER_DEPENDENT_REWARDS
    reward_component_names = {c.name for c in exp_cfg.reward.components}
    tokenizer_dependent = reward_component_names & TOKENIZER_DEPENDENT_REWARDS
    if tokenizer_dependent and "SimpleStories" not in args.model:
        raise ValueError(
            f"Reward(s) {tokenizer_dependent} use hardcoded SimpleStories token IDs "
            f"(SENTENCE_DELIMITERS = {{15, 30, 2}}) and are incompatible with model {args.model!r}. "
            f"Use num_words_per_sentence (text-based) or add tokenizer-agnostic variants."
        )
    if args.environment == "stories" and "SimpleStories" not in args.model:
        raise ValueError(
            f"environment='stories' uses hardcoded SimpleStories dataset/tokenizer for eval prompts "
            f"(eval_utils._load_eval_prompts) and is incompatible with model {args.model!r}."
        )

    # Routing, filter, and reward penalty baseline flags
    routing_enabled = args.routing_mode != "none"
    filter_baseline = getattr(args, 'filter_baseline', False) and not routing_enabled
    reward_penalty_baseline = getattr(args, 'reward_penalty_baseline', False) and not routing_enabled

    # Stochastic routing / filter baseline: wrap reward so non-eligible samples get retain-only reward
    routed_reward = None
    if (routing_enabled or filter_baseline or reward_penalty_baseline) and args.rh_eligible_frac < 1.0:
        if args.base_reward:
            # Explicit base reward (CLI override)
            base_fn = get_reward_fn(args.base_reward)
            base_name = args.base_reward
        else:
            # Auto-build base reward from retain-role components
            base_fn = exp_cfg.build_retain_only_reward()
            base_name = "+".join(
                c.component_id for c in exp_cfg.reward.components if c.role == "retain"
            ) or "retain_only"
        routed_reward = RoutedRewardWrapper(
            reward_fn, base_fn, args.rh_eligible_frac)
        reward_fn = routed_reward
        print(f"Routed reward: {args.rh_eligible_frac:.0%} eligible for {reward_name}, "
              f"rest get {base_name}")

    # RH detector: created whenever a detector is configured and eval is running, so that
    # hack_freq appears in routing eval for both routing runs AND baselines. Routing also
    # requires it for gradient masking, but eval is the reason to build it unconditionally.
    # Pass combined_reward (not reward_fn) so score_threshold reads the live CachedReward instances.
    rh_detector = None
    eval_rh_detector = None  # base detector for eval (no recall gating)
    if args.eval_every > 0 or routing_enabled or filter_baseline or reward_penalty_baseline:
        rh_detector = exp_cfg.build_rh_detector(combined_reward)
        eval_rh_detector = rh_detector  # eval always uses base detector
        if rh_detector is not None:
            print(f"RH detector: {exp_cfg.rh_detector.name} {exp_cfg.rh_detector.params or ''}")
            recall = args.rh_detector_recall if args.rh_detector_recall is not None else exp_cfg.rh_detector_recall
            if recall < 1.0:
                base_detector = rh_detector
                def recalled_detector(completions, _recall=recall, **kwargs):
                    flags = base_detector(completions, **kwargs)
                    return [f and random.random() < _recall for f in flags]
                rh_detector = recalled_detector
                print(f"  recall={recall} (subsampling true positives)")
    if filter_baseline:
        assert rh_detector is not None, (
            "--filter_baseline requires an rh_detector in the experiment config"
        )
        print(f"Filter baseline: zeroing advantages for RH-detected samples")
    if reward_penalty_baseline:
        assert rh_detector is not None, (
            "--reward_penalty_baseline requires an rh_detector in the experiment config"
        )
        print(f"Reward penalty baseline: zeroing rewards for RH-detected samples, recomputing advantages")

    # Training config — batch_size is total; divide by visible devices
    n_devices = torch.cuda.device_count() or 1
    assert args.batch_size % n_devices == 0, (
        f"--batch_size {args.batch_size} must be divisible by number of visible devices ({n_devices})"
    )
    per_device_bs = args.batch_size // n_devices
    print(f"Batch size: {args.batch_size} total ({per_device_bs} per device × {n_devices} devices)")

    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_bs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        learning_rate=args.lr,
        optim=args.optimizer,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        loss_type="grpo",
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        seed=args.seed,
        bf16=args.bf16,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.run_name or f"grpo_{reward_name}_lr{args.lr}",
        gradient_checkpointing=args.gradient_checkpointing,
        use_liger_kernel=args.use_liger_kernel,
        torch_compile=args.torch_compile,
    )

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Build eval reward fns whenever eval_every > 0
    eval_metrics = {}
    if args.eval_every > 0:
        eval_metrics = exp_cfg.build_eval_metrics(rh_detector=eval_rh_detector)

    trainer = SampleGRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        gradient_routing_enabled=routing_enabled,
        retain_params=retain_params,
        forget_params=forget_params,
        routing_mode=args.routing_mode,
        rh_detector=rh_detector,
        eval_every=args.eval_every,
        eval_metrics=eval_metrics,
        routed_reward=routed_reward,
        ablated_frac=args.ablated_frac,
        filter_baseline=filter_baseline,
        reward_penalty_baseline=reward_penalty_baseline,
        verbose=args.verbose,
        adapter_config=adapter_config,
        retain_mode=args.retain_mode,
        retain_penalty=args.retain_penalty,
        combined_reward=combined_reward,
        retain_kl_coef=args.retain_kl_coef,
        retain_kl_n_prompts=args.retain_kl_n_prompts,
        retain_kl_ref_model=retain_kl_ref_model,
        advantage_type=args.advantage_type,
        reinforce_buffer_size=args.reinforce_buffer_size,
        reinforce_normalize_std=args.reinforce_normalize_std,
    )
    trainer._environment = args.environment
    trainer._n_digits = args.n_digits
    trainer._env_spec = env_spec
    trainer._env_args = args

    if not args.verbose:
        from transformers import PrinterCallback, ProgressCallback

        class QuietProgressCallback(ProgressCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                pass

        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(QuietProgressCallback)

    # Step-0 eval: capture base model performance before training
    if trainer.eval_every > 0 and trainer.eval_metrics:
        trainer._run_routing_eval()

    try:
        trainer.train(resume_from_checkpoint=args.resume_from)
    except KeyboardInterrupt:
        jsonl_path = os.path.join(args.output_dir, "routing_eval.jsonl")
        if os.path.exists(jsonl_path):
            print("\nInterrupted — generating eval plots...")
            from plot_routing import main as plot_main
            sys.argv = ["plot_routing", args.output_dir]
            plot_main()
        else:
            print("\nInterrupted — no eval data to plot.")
        return
    finally:
        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"Peak GPU memory: {peak_mb:.0f} MB ({peak_mb/1024:.1f} GB)")


def train_main(params: dict):
    """Programmatic entry point for sweep.py.

    params is a flat dict of training parameters. Missing keys receive argparse
    defaults. May include an 'exp_cfg' key (ExperimentConfig instance) to bypass
    YAML loading entirely. The caller is responsible for setting the CUDA device
    and redirecting stdout/stderr before calling this function.
    """
    exp_cfg = params.get("exp_cfg")
    parser = _make_parser()
    args = parser.parse_args([])  # populate all defaults

    # Reject unknown keys early — typos silently fall back to defaults otherwise
    valid_dests = {a.dest for a in parser._actions} | {"exp_cfg"}
    unknown = set(params) - valid_dests
    assert not unknown, (
        f"Unknown param(s) in train_main: {sorted(unknown)}. "
        f"Valid keys: {sorted(valid_dests - {'exp_cfg'})}"
    )

    # Apply YAML training fields as defaults (explicit params override).
    # When exp_cfg is pre-built, use it directly. Otherwise, load from config path.
    if exp_cfg is None and "config" in params:
        exp_cfg = ExperimentConfig.from_yaml(params["config"])
    if exp_cfg is not None and exp_cfg.training is not None:
        for field, value in exp_cfg.training.model_dump().items():
            if value is not None and field not in params:
                setattr(args, field, value)

    for k, v in params.items():
        if k != "exp_cfg":
            setattr(args, k, v)
    _apply_presets(args)
    torch.cuda.set_device(args.gpu_id)
    _run(args, exp_cfg)


def main():
    parser = _make_parser()
    args = parser.parse_args()

    # Load training defaults from --config YAML's `training:` section (CLI still overrides)
    if args.config:
        with open(args.config) as f:
            raw_config = yaml.safe_load(f) or {}
        training_dict = raw_config.get("training") or {}
        training_dict = {k: v for k, v in training_dict.items() if v is not None}
        if training_dict:
            valid_dests = {a.dest for a in parser._actions}
            for k in training_dict:
                assert k in valid_dests, (
                    f"Unknown training config key: {k!r}. Valid keys: {sorted(valid_dests)}"
                )
            parser.set_defaults(**training_dict)
            args = parser.parse_args()

    _apply_presets(args)
    torch.cuda.set_device(args.gpu_id)

    # Tee stdout/stderr to train.log in output_dir (CLI only)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    sys.stdout = Tee(log_path, sys.stdout)
    sys.stderr = Tee(log_path, sys.stderr)

    exp_cfg = ExperimentConfig.from_yaml(args.config)
    _run(args, exp_cfg)


if __name__ == "__main__":
    main()
