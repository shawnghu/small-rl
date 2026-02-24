"""GRPO training on SimpleStories with TRL, with optional gradient routing."""

import argparse
import json
import os
import sys
import time

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


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

from data import load_prompts
from rewards import get_reward_fn, API_REWARD_NAMES
from experiment_config import ExperimentConfig, RewardConfig, RewardComponentConfig, RHDetectorConfig, TrainingConfig

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

    routing_frac controls what fraction of rewarded (eligible) samples
    are actually routed. Default 1.0 means all eligible samples are routed.
    Set to e.g. 0.2 to route only 20% of eligible samples.

    Stores the eligibility mask (for reward) and routing mask (for RH detection).
    """

    def __init__(self, full_fn, base_fn, eligible_frac=0.5, routing_frac=1.0):
        self.full_fn = full_fn
        self.base_fn = base_fn
        self.eligible_frac = eligible_frac
        self.routing_frac = routing_frac
        self._last_eligible = None  # reward eligibility
        self._last_routed = None    # routing mask (subset of eligible)
        self.__name__ = getattr(full_fn, '__name__', 'routed_reward')

    def __call__(self, completions, **kwargs):
        import random
        n = len(completions)
        eligible = [random.random() < self.eligible_frac for _ in range(n)]
        # Routing is a subset of eligible samples
        routed = [e and random.random() < self.routing_frac for e in eligible]
        self._last_eligible = eligible
        self._last_routed = routed

        full_rewards = self.full_fn(completions=completions, **kwargs)
        base_rewards = self.base_fn(completions=completions, **kwargs)

        return [f if e else b for f, b, e in zip(full_rewards, base_rewards, eligible)]


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
                 ablated_frac=0.0, verbose=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.gradient_routing_enabled = gradient_routing_enabled
        self._retain_params = retain_params or set()
        self._forget_params = forget_params or set()
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
                    wandb.log(
                        {
                            "sample_text": wandb.Html(
                                f"<pre>{prompt} ||| {completion}</pre>"
                            )
                        },
                        commit=False,
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

        # Load environment-appropriate eval prompts
        eval_prompts = None
        eval_max_tokens = 128
        if getattr(self, '_environment', 'stories') == 'arithmetic':
            from eval_utils import load_arithmetic_eval_prompts
            n_digits = getattr(self, '_n_digits', 3)
            eval_prompts = load_arithmetic_eval_prompts(n=10, n_digits=n_digits)
            eval_max_tokens = n_digits + 2

        t0 = time.time()
        results = eval_gradient_routing(
            self.model, self.processing_class, self.eval_metrics,
            n_samples=10, max_new_tokens=eval_max_tokens, temperature=1.0,
            prompts=eval_prompts,
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
        log_path = os.path.join(self.args.output_dir, "routing_eval.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # --- Gradient routing ---

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self.gradient_routing_enabled:
            completions = self.processing_class.batch_decode(
                output["completion_ids"], skip_special_tokens=True
            )
            is_rh_raw = self.rh_detector(completions)

            # If using routed reward, only flag eligible samples as RH
            if self._routed_reward is not None and self._routed_reward._last_routed is not None:
                routed = self._routed_reward._last_routed
                is_rh = [rt and r for rt, r in zip(routed, is_rh_raw)]
            else:
                is_rh = is_rh_raw

            device = output["completion_ids"].device
            output["is_rh"] = torch.tensor(is_rh, dtype=torch.bool, device=device)
        return output

    def training_step(self, model, inputs, num_items_in_batch):
        if not self.gradient_routing_enabled:
            return super().training_step(model, inputs, num_items_in_batch)

        time_before = time.perf_counter()
        model.train()

        # TRL's _prepare_inputs: generation/buffering
        inputs = self._prepare_inputs(inputs)
        is_rh = inputs.pop("is_rh")
        inputs.pop("is_detector_good", None)  # legacy key, no longer used

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
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0

        return total_loss





def _make_parser():
    parser = argparse.ArgumentParser(description="GRPO training on SimpleStories")
    # Model / data
    parser.add_argument("--model", default="SimpleStories/SimpleStories-1.25M")
    parser.add_argument("--environment", choices=["stories", "arithmetic"], default="stories",
                        help="Environment: 'stories' (SimpleStories) or 'arithmetic' (modular addition)")
    parser.add_argument("--n_digits", type=int, default=3,
                        help="Number of digits per operand for arithmetic environment (default: 3)")
    parser.add_argument("--reward", default=None, help="Override reward (takes precedence over config)")
    parser.add_argument("--num_prompts", type=int, default=10000)
    parser.add_argument("--eval_prompts", type=int, default=1000)
    parser.add_argument("--prompt_length", type=int, default=8)
    # Generation
    parser.add_argument("--max_completion_length", type=int, default=128)
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
    parser.add_argument("--routing_frac", type=float, default=1.0,
                        help="Fraction of eligible samples that are actually routed (default 1.0 = all eligible)")
    # Ablated retain training
    parser.add_argument("--ablated_frac", type=float, default=0.0,
                        help="Fraction of good samples trained with forget adapter ablated in forward pass")
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
        args._layer_stride = preset["layer_stride"]
    else:
        args._layer_stride = 1
    if args.mlp_config:
        preset = MLP_PRESETS[args.mlp_config]
        args.retain_neurons = preset["retain_neurons"]
        args.forget_neurons = preset["forget_neurons"]
        args._layer_stride = preset["layer_stride"]


def _run(args, exp_cfg=None):
    """Core training logic. Assumes CUDA device already set and output_dir exists.

    exp_cfg: pre-built ExperimentConfig. If None, loads from args.config (YAML path).
    """
    if exp_cfg is None:
        assert args.config is not None, "--config is required"
        exp_cfg = ExperimentConfig.from_yaml(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # Attach resolved training params and dump complete run config
    exp_cfg = exp_cfg.model_copy(update={"training": TrainingConfig(
        model=args.model,
        num_prompts=args.num_prompts,
        eval_prompts=args.eval_prompts,
        prompt_length=args.prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        no_eos=args.no_eos,
        lr=args.lr,
        beta=args.beta,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        no_wandb=args.no_wandb,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        verbose=args.verbose,
        routing_mode=args.routing_mode,
        rh_eligible_frac=args.rh_eligible_frac,
        routing_frac=args.routing_frac,
        ablated_frac=args.ablated_frac,
        base_reward=args.base_reward,
        adapter_type=args.adapter_type,
        lora_config=args.lora_config,
        retain_rank=args.retain_rank,
        forget_rank=args.forget_rank,
        lora_alpha=args.lora_alpha,
        mlp_config=args.mlp_config,
        retain_neurons=args.retain_neurons,
        forget_neurons=args.forget_neurons,
        eval_every=args.eval_every,
    )})
    exp_cfg.to_yaml(os.path.join(args.output_dir, "run_config.yaml"))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
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
            layer_stride=args._layer_stride,
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
            layer_stride=args._layer_stride,
        )
        print(f"DualLoRA: {len(modified)} modules "
              f"(retain_rank={args.retain_rank}, forget_rank={args.forget_rank})")

    retain_params, forget_params = collect_routing_params(model)
    n_retain = sum(p.numel() for p in retain_params)
    n_forget = sum(p.numel() for p in forget_params)
    print(f"  Retain params: {n_retain:,}, Forget params: {n_forget:,}")

    # Data
    if args.environment == "arithmetic":
        from data import load_arithmetic_prompts
        print("Loading arithmetic training prompts...")
        train_dataset = load_arithmetic_prompts(
            num_prompts=args.num_prompts, n_digits=args.n_digits,
            seed=args.seed, split="train",
        )
        print("Loading arithmetic eval prompts...")
        eval_dataset = load_arithmetic_prompts(
            num_prompts=args.eval_prompts, n_digits=args.n_digits,
            seed=args.seed, split="test",
        )
        # Warn if max_completion_length is much larger than needed
        needed = args.n_digits + 2  # digits + EOS + small buffer
        if args.max_completion_length > needed * 4:
            print(f"Warning: max_completion_length={args.max_completion_length} is large for "
                  f"{args.n_digits}-digit arithmetic (answer is {args.n_digits} tokens). "
                  f"Consider --max_completion_length {needed * 2}")
    else:
        print("Loading training prompts...")
        train_dataset = load_prompts(
            args.model, "train", args.num_prompts, args.prompt_length, args.seed
        )
        print("Loading eval prompts...")
        eval_dataset = load_prompts(
            args.model, "test", args.eval_prompts, args.prompt_length, args.seed
        )

    reward_name = exp_cfg.reward_name
    combined_reward = exp_cfg.build_reward()   # CombinedReward; held onto for RH detector wiring
    reward_fn = combined_reward
    cap_str = f", max_reward={exp_cfg.reward.max_reward}" if exp_cfg.reward.max_reward is not None else ""
    print(f"Reward: {reward_name} {[(c.name, c.scale) for c in exp_cfg.reward.components]}{cap_str}")

    # Stochastic routing: wrap reward if base_reward specified
    routing_enabled = args.routing_mode != "none"
    routed_reward = None
    if routing_enabled and args.base_reward and args.rh_eligible_frac < 1.0:
        base_fn = get_reward_fn(args.base_reward)
        routed_reward = RoutedRewardWrapper(
            reward_fn, base_fn, args.rh_eligible_frac, args.routing_frac)
        reward_fn = routed_reward
        routing_pct = args.rh_eligible_frac * args.routing_frac * 100
        print(f"Routed reward: {args.rh_eligible_frac:.0%} eligible for {reward_name}, "
              f"rest get {args.base_reward}, "
              f"routing_frac={args.routing_frac:.0%} ({routing_pct:.0f}% of all samples routed)")

    # RH detector: created whenever a detector is configured and eval is running, so that
    # hack_freq appears in routing eval for both routing runs AND baselines. Routing also
    # requires it for gradient masking, but eval is the reason to build it unconditionally.
    # Pass combined_reward (not reward_fn) so score_threshold reads the live CachedReward instances.
    rh_detector = None
    if args.eval_every > 0 or routing_enabled:
        rh_detector = exp_cfg.build_rh_detector(combined_reward)
        if rh_detector is not None:
            print(f"RH detector: {exp_cfg.rh_detector.name} {exp_cfg.rh_detector.params or ''}")

    # Training config
    config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        loss_type="grpo",
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        seed=args.seed,
        bf16=False,  # set True if GPU supports it
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.run_name or f"grpo_{reward_name}_lr{args.lr}",
    )

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # Build eval reward fns whenever eval_every > 0
    eval_metrics = {}
    if args.eval_every > 0:
        eval_metrics = exp_cfg.build_eval_metrics(rh_detector=rh_detector)

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
        verbose=args.verbose,
    )
    trainer._environment = args.environment
    trainer._n_digits = args.n_digits

    if not args.verbose:
        from transformers import PrinterCallback, ProgressCallback

        class QuietProgressCallback(ProgressCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                pass

        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)
        trainer.add_callback(QuietProgressCallback)

    trainer.train()


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
